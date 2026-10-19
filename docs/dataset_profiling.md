# Dataset profiling

Terminal (downloads data on first use; does not train):

```bash
egbench profile --dataset cora
egbench profile --dataset amazon-ratings --output experiments/amazon-profile.json
egbench profile --dataset custom --dataset-path data/Mydata/graph.npz
egbench profile --help
```

Python/notebook:

```python
from efficientgraphbench.datasets import profile_dataset

profile = profile_dataset("cora")
print(profile)
details = profile.to_dict()
profile.save_json("experiments/cora-profile.json")
```

Already loaded a dataset? Call `print(dataset.profile())`. Custom data accepts the same loading options as `load_dataset`, including `dataset_path` or named `nodes` and `edges` CSV files. A profile computes all statistics from the actual loaded graph; example numbers are not hard-coded.

Definitions:

- **Edges (stored):** edge-index column count, including duplicates and self-loops. Undirected PyG datasets usually store both directions of a connection.
- **Density:** unique non-self directed arcs divided by N(N−1). For symmetric graphs this equals the usual undirected density. Zero for N=1.
- **Degree:** number of unique outgoing neighbors, excluding self-loops and duplicate arcs; for symmetric graphs, the ordinary neighbor count. Includes isolated nodes in averages and medians. Directed graphs explicitly say “outgoing”.
- **Homophily:** fraction of unique non-self arcs connecting equal class labels, using only edges whose endpoints are both labeled. `-1` labels are excluded. N/A when there are no eligible edges. This is edge homophily, not adjusted homophily.
- **Class imbalance:** largest/smallest labeled class count over all declared classes. Heuristic labels: low ≤1.5, moderate >1.5 and ≤3, high >3 or a missing class. No labeled nodes means unavailable. JSON includes the class counts and ratio (null when a class count is zero). These categories are project conventions, not a universal standard.
- **Components:** weakly connected components for directed graphs, ordinary connected components for undirected graphs, including isolated nodes. Largest component percentage uses all nodes.
- **Feature sparsity:** percentage of feature entries exactly equal to zero. Uses retained `raw_x` when available, otherwise loaded `x`; JSON records which and the loader preprocessing. It does not measure text missingness or missing values.

Profiles also save isolated-node counts, unlabeled-node counts, split identity and the dataset fingerprint. Label statistics describe the whole dataset; they are not used to change training splits or select models.

Profiling uses a sparse adjacency, with O(N+E) storage, and scans features in chunks. It needs CPU memory for the loaded graph plus sparse analysis structures; it is not a streaming loader for graphs that exceed RAM. It does not build a dense N×N matrix or run automatically inside timed training benchmarks.
