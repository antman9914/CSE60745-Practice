"""Dataset preparation for the graph-mining lab (LastFM Asia social network).

This script is run ONCE by the instructor. It turns the raw SNAP download into the
files that students load in the two lab notebooks.

Source
------
https://snap.stanford.edu/data/feather-lastfm-social.html
Download: https://snap.stanford.edu/data/lastfm_asia.zip
(note: the archive extracts into a directory named "lasftm_asia", a typo upstream)

The graph has 7,624 nodes and 27,806 undirected edges. Node features and the
artist-liked features are deliberately NOT used: both lab tasks are purely
structural. The `target` column (country, 18 classes) is exported but is not used
as a prediction target anywhere in the lab.

Outputs
-------
The two tasks get their own directory and share no files, so each notebook downloads
only what it needs.

data/task1/
    lp_graph_obs.csv  src,dst                     the observable graph (train positives)
    lp_train.csv      query_id,src,dst,label      1 positive + 5 negatives per query
    lp_val.csv        query_id,src,dst,label      1 positive + 20 negatives per query
    lp_test.csv       query_id,src,dst,label      1 positive + 20 negatives per query
    split_stats.json  summary statistics of the split

data/task2/
    edges_all.csv     src,dst                     full undirected graph
    node_country.csv  node_id,country             country label, for exploration only

Link-prediction protocol
------------------------
Edges are split 70/15/15 into train/val/test. Only the train positives form the
observable graph `lp_graph_obs.csv`; val and test positive edges are absent from it.

Negatives are sampled *per query*, not globally: for a positive edge (u, v) one
endpoint is picked at random as the query source `src`, and K negative destinations
are drawn uniformly from nodes that are NOT connected to `src` anywhere in the full
graph. This makes each query a ranking problem over 1 + K candidates, which is what
Hit@1 and MRR require. Rows are shuffled within each query so that the position of
the positive carries no information.

Usage
-----
    python prepare_data.py
"""

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "lasftm_asia"
T1_DIR = ROOT / "data" / "task1"
T2_DIR = ROOT / "data" / "task2"

SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder, so the three fractions always sum to exactly 1.0

NEG_PER_POS_TRAIN = 5    # training set: 1 positive + 5 negatives
NEG_PER_POS_EVAL = 20    # val / test:   1 positive + 20 negatives


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_raw():
    """Read the raw SNAP CSVs and return (undirected edge list, node->country series)."""
    edges_df = pd.read_csv(RAW_DIR / "lastfm_asia_edges.csv")
    target_df = pd.read_csv(RAW_DIR / "lastfm_asia_target.csv")

    # Canonicalise every edge as an ordered tuple (small, large), drop self-loops and
    # duplicates. The raw file is already clean, but we do not want to rely on that.
    edges = set()
    for u, v in edges_df[["node_1", "node_2"]].itertuples(index=False):
        u, v = int(u), int(v)
        if u == v:
            continue
        edges.add((u, v) if u < v else (v, u))

    country = target_df.set_index("id")["target"].sort_index()
    return sorted(edges), country


# --------------------------------------------------------------------------------------
# Splitting and negative sampling
# --------------------------------------------------------------------------------------

def split_edges(edges, rng):
    """Split the edge list 70/15/15 uniformly at random.

    No constraint is imposed on connectivity: nodes are allowed to become isolated in
    the training graph. That cold-start situation is intentional and is one of the
    discussion points of the lab.
    """
    perm = rng.permutation(len(edges))
    n_train = int(round(TRAIN_FRAC * len(edges)))
    n_val = int(round(VAL_FRAC * len(edges)))

    train = [edges[i] for i in perm[:n_train]]
    val = [edges[i] for i in perm[n_train:n_train + n_val]]
    test = [edges[i] for i in perm[n_train + n_val:]]
    return train, val, test


def build_queries(pos_edges, forbidden, n_nodes, n_neg, rng, query_id_start=0):
    """Turn positive edges into ranking queries with sampled negatives.

    Parameters
    ----------
    pos_edges : list of (u, v)
        Positive edges of one split.
    forbidden : set of (small, large)
        Every edge of the FULL graph. A negative must not be a real edge in any split,
        otherwise we would label a true friendship as a non-friendship.
    n_nodes : int
        Number of nodes; negative destinations are drawn uniformly from range(n_nodes).
    n_neg : int
        Number of negatives per positive.

    Returns
    -------
    pandas.DataFrame with columns query_id, src, dst, label.
    """
    records = []
    for offset, (u, v) in enumerate(pos_edges):
        qid = query_id_start + offset

        # Pick which endpoint plays the role of "the user asking for recommendations".
        # Randomising this avoids any systematic bias towards low node ids.
        src, dst = (u, v) if rng.random() < 0.5 else (v, u)

        rows = [(qid, src, dst, 1)]
        drawn = set()
        while len(drawn) < n_neg:
            for w in rng.integers(0, n_nodes, size=n_neg + 8):
                w = int(w)
                if w == src or w in drawn:
                    continue
                if (min(src, w), max(src, w)) in forbidden:
                    continue
                drawn.add(w)
                if len(drawn) == n_neg:
                    break
        rows.extend((qid, src, w, 0) for w in sorted(drawn))

        # Shuffle inside the query so the positive is not always the first row.
        rng.shuffle(rows)
        records.extend(rows)

    return pd.DataFrame(records, columns=["query_id", "src", "dst", "label"])


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def split_statistics(graph_obs, splits, queries):
    """Collect the numbers worth reporting (and worth discussing) about the split."""
    degrees = np.array([d for _, d in graph_obs.degree()])
    isolated = set(n for n, d in graph_obs.degree() if d == 0)

    stats = {
        "n_nodes": graph_obs.number_of_nodes(),
        "n_edges_full_graph": sum(len(e) for e in splits.values()),
        "split_sizes": {name: len(e) for name, e in splits.items()},
        "observable_graph": {
            "n_edges": graph_obs.number_of_edges(),
            "n_isolated_nodes": len(isolated),
            "frac_isolated_nodes": round(len(isolated) / graph_obs.number_of_nodes(), 4),
            "mean_degree": round(float(degrees.mean()), 3),
            "median_degree": int(np.median(degrees)),
            "max_degree": int(degrees.max()),
            "n_connected_components": nx.number_connected_components(graph_obs),
        },
        "queries": {},
    }

    for name, df in queries.items():
        pos = df[df["label"] == 1]
        cold = pos["src"].isin(isolated)
        stats["queries"][name] = {
            "n_queries": int(len(pos)),
            "n_rows": int(len(df)),
            "candidates_per_query": int(len(df) // len(pos)),
            # Queries whose source node has no edge at all in the observable graph:
            # every candidate looks identical to any purely structural feature.
            "n_cold_start_queries": int(cold.sum()),
            "frac_cold_start_queries": round(float(cold.mean()), 4),
        }
    return stats


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    T1_DIR.mkdir(parents=True, exist_ok=True)
    T2_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    edges, country = load_raw()
    n_nodes = int(country.index.max()) + 1
    forbidden = set(edges)
    print(f"loaded {len(edges)} undirected edges over {n_nodes} nodes")

    # ---- Task 2 inputs: the complete graph, plus the country label ---------------------
    pd.DataFrame(edges, columns=["src", "dst"]).to_csv(T2_DIR / "edges_all.csv", index=False)
    country.rename("country").rename_axis("node_id").to_csv(T2_DIR / "node_country.csv")

    # ---- Task 1 inputs: split, observable graph, ranking queries -----------------------
    train_e, val_e, test_e = split_edges(edges, rng)
    splits = {"train": train_e, "val": val_e, "test": test_e}
    print("split sizes:", {k: len(v) for k, v in splits.items()})

    graph_obs = nx.Graph()
    graph_obs.add_nodes_from(range(n_nodes))
    graph_obs.add_edges_from(train_e)
    pd.DataFrame(train_e, columns=["src", "dst"]).to_csv(
        T1_DIR / "lp_graph_obs.csv", index=False
    )

    # `random.Random` is used for shuffling rows inside a query; numpy handles the draws.
    py_rng = random.Random(SEED)

    class _Rng:
        """Small adapter so build_queries can use both generators through one object."""

        def random(self):
            return py_rng.random()

        def shuffle(self, seq):
            py_rng.shuffle(seq)

        def integers(self, low, high, size):
            return rng.integers(low, high, size=size)

    adapter = _Rng()

    queries = {}
    qid = 0
    for name, pos_edges in splits.items():
        n_neg = NEG_PER_POS_TRAIN if name == "train" else NEG_PER_POS_EVAL
        df = build_queries(pos_edges, forbidden, n_nodes, n_neg, adapter, query_id_start=qid)
        qid += len(pos_edges)
        df.to_csv(T1_DIR / f"lp_{name}.csv", index=False)
        queries[name] = df
        print(f"  {name}: {len(pos_edges)} queries x (1 + {n_neg}) = {len(df)} rows")

    # ---- Sanity checks -----------------------------------------------------------------
    for name, df in queries.items():
        assert df.groupby("query_id")["label"].sum().eq(1).all(), f"{name}: not exactly 1 positive"
        assert not df.duplicated(["query_id", "dst"]).any(), f"{name}: duplicate candidate"
        neg = df[df["label"] == 0]
        clash = [(u, v) for u, v in neg[["src", "dst"]].itertuples(index=False)
                 if (min(u, v), max(u, v)) in forbidden]
        assert not clash, f"{name}: {len(clash)} negatives are real edges"
    pos_pairs = {name: {(min(u, v), max(u, v))
                        for u, v in df[df["label"] == 1][["src", "dst"]].itertuples(index=False)}
                 for name, df in queries.items()}
    assert not (pos_pairs["val"] | pos_pairs["test"]) & set(train_e), "eval edge leaked into graph"
    print("sanity checks passed")

    # ---- Statistics ---------------------------------------------------------------------
    stats = split_statistics(graph_obs, splits, queries)
    with open(T1_DIR / "split_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2))
    for d in (T1_DIR, T2_DIR):
        print(f"\nwrote {len(list(d.glob('*')))} files to {d}: "
              f"{', '.join(sorted(p.name for p in d.iterdir()))}")


if __name__ == "__main__":
    main()
