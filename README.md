# Hands-on Practice - Social Network Analysis

The goal of this practice is to find out how far the graph properties from the lectures can actually help with real-world applications: friend recommendation and community detection in social network analysis.

**Dataset.** LastFM Asia Social Network consists of 7,624 LastFM users from Asian countries and 27,806 mutual-follower edges, distributed by
[SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html).

| notebook | task | TODO |
|---|---|---|---|
| `task1_link_prediction.ipynb` | Friend recommendation | pair features, model, scoring | 
| `task2_community_detection.ipynb` | Community detection | the label-update scoring function | 


## Data files

The first cell of each notebook will automatically download needed data.

**Task 1** — `data/task1/`

| file | contents |
|---|---|
| `lp_graph_obs.csv` | `src,dst` — the observable graph (70% of edges) |
| `lp_train.csv` | `query_id,src,dst,label` — 19,464 queries, 1 positive + 5 negatives |
| `lp_val.csv` | 4,171 queries, 1 positive + 20 negatives |
| `lp_test.csv` | 4,171 queries, 1 positive + 20 negatives |
| `split_stats.json` | statistics of the split |

**Task 2** — `data/task2/`

| file | contents |
|---|---|
| `edges_all.csv` | `src,dst` — the complete graph, all 27,806 edges |
| `node_country.csv` | `node_id,country` — 18 classes, for the open-ended section only |

In Task 1, all rows sharing a `query_id` form one ranking problem with exactly one
correct answer.


