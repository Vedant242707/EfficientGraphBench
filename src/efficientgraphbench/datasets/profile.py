"""Exact sparse graph profiling, independent of model training."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class DatasetProfile:
    metrics: dict

    def to_dict(self):
        return dict(self.metrics)

    def __getitem__(self, key):
        return self.metrics[key]

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metrics, indent=2, allow_nan=False), encoding="utf-8")
        return path

    def __str__(self):
        m = self.metrics
        rows = [
            ("Nodes", f"{m['num_nodes']:,}"),
            ("Edges (stored)", f"{m['num_edges']:,}"),
            ("Features", f"{m['num_features']:,}"),
            ("Classes", str(m["num_classes"])),
            ("Graph density", f"{m['graph_density']:.6f}"),
            ("Avg degree", f"{m['avg_degree']:.2f}"),
            ("Median degree", f"{m['median_degree']:g}"),
            ("Max degree", f"{m['max_degree']:,}"),
            ("Homophily", "N/A" if m["edge_homophily"] is None else f"{m['edge_homophily']:.4f}"),
            (
                "Class imbalance",
                m["class_imbalance"]
                + (
                    f" ({m['class_imbalance_ratio']:.2f}x)"
                    if m["class_imbalance_ratio"] is not None
                    else ""
                ),
            ),
            ("Connected comps", f"{m['connected_components']:,}"),
            ("Largest comp", f"{m['largest_component_pct']:.2f}%"),
            ("Feature sparsity", f"{m['feature_sparsity_pct']:.2f}%"),
        ]
        return (
            f"Dataset: {m['dataset']}\n\n"
            + "\n".join(f"{k + ':':22}{v}" for k, v in rows)
            + f"\n\nDegree: {m['degree_definition']}; components: weak/undirected."
            + "\nDensity, degrees and homophily exclude duplicate edges and self-loops."
            + "\nClass imbalance uses max/min labeled class counts: "
            "low <=1.5, moderate <=3, high >3."
        )


def profile_dataset(dataset, root="data", **options):
    """Profile a built-in/custom dataset name or an already loaded DatasetBundle.

    Uses a sparse adjacency (O(N+E) storage), never a dense N-by-N matrix.
    Degree means unique outgoing neighbors; on symmetric graphs this is degree.
    Labels are used descriptively only, never to alter benchmark input or splits.
    """
    from efficientgraphbench.datasets.loaders import load_dataset
    from efficientgraphbench.datasets.registry import DatasetBundle

    if isinstance(dataset, DatasetBundle):
        if options:
            raise ValueError("Load options cannot be supplied with an existing DatasetBundle")
        bundle = dataset
    else:
        bundle = load_dataset(dataset, root=root, **options)
    data = bundle.data
    n = int(data.num_nodes)
    if n < 1:
        raise ValueError("Cannot profile an empty graph")
    edges = data.edge_index.detach().cpu().numpy()
    adjacency = csr_matrix(
        (np.ones(edges.shape[1], dtype=bool), (edges[0], edges[1])), shape=(n, n)
    )
    adjacency.setdiag(False)
    adjacency.eliminate_zeros()
    degrees = np.diff(adjacency.indptr)
    symmetric = (adjacency != adjacency.T).nnz == 0
    count, components = connected_components(adjacency, directed=True, connection="weak")
    sizes = np.bincount(components)
    labels = data.y.detach().cpu().numpy()
    classes = np.bincount(labels[labels >= 0], minlength=bundle.num_classes)
    ratio = float(classes.max() / classes.min()) if classes.min() > 0 else None
    imbalance = (
        "unavailable"
        if not classes.sum()
        else "high"
        if ratio is None or ratio > 3
        else "moderate"
        if ratio > 1.5
        else "low"
    )
    eligible, same = 0, 0
    # Bound temporary endpoint/label arrays, even on multi-million-edge graphs.
    for start in range(0, adjacency.nnz, 1_000_000):
        stop = min(start + 1_000_000, adjacency.nnz)
        source = np.searchsorted(adjacency.indptr, np.arange(start, stop), side="right") - 1
        left, right = labels[source], labels[adjacency.indices[start:stop]]
        valid = (left >= 0) & (right >= 0)
        eligible += int(valid.sum())
        same += int(((left == right) & valid).sum())
    features = getattr(data, "raw_x", None)
    feature_source = "raw_x" if features is not None else "x"
    features = features if features is not None else data.x
    nonzero = 0
    rows_per_chunk = max(1, 1_000_000 // features.shape[1])
    for chunk in features.split(rows_per_chunk):
        nonzero += int(torch.count_nonzero(chunk).item())
    return DatasetProfile(
        {
            "dataset": bundle.name,
            "num_nodes": n,
            "num_edges": int(data.num_edges),
            "num_features": int(data.num_node_features),
            "num_classes": bundle.num_classes,
            "unique_nonself_arcs": int(adjacency.nnz),
            "is_undirected": symmetric,
            "graph_density": float(adjacency.nnz / (n * (n - 1))) if n > 1 else 0.0,
            "avg_degree": float(degrees.mean()),
            "median_degree": float(np.median(degrees)),
            "max_degree": int(degrees.max()),
            "degree_definition": "unique neighbors" if symmetric else "unique outgoing neighbors",
            "edge_homophily": same / eligible if eligible else None,
            "homophily_labeled_arcs": eligible,
            "class_counts": classes.tolist(),
            "unlabeled_nodes": int((labels < 0).sum()),
            "class_imbalance_ratio": ratio,
            "class_imbalance": imbalance,
            "class_imbalance_thresholds": {"low_max": 1.5, "moderate_max": 3},
            "connected_components": int(count),
            "largest_component_nodes": int(sizes.max()),
            "largest_component_pct": float(sizes.max() / n * 100),
            "isolated_nodes": int(
                ((degrees == 0) & (np.asarray(adjacency.sum(axis=0)).ravel() == 0)).sum()
            ),
            "feature_sparsity_pct": float((1 - nonzero / features.numel()) * 100),
            "feature_source": feature_source,
            "feature_preprocessing": bundle.preprocessing,
            "split": bundle.split,
            "edge_policy": bundle.edge_policy,
            "dataset_fingerprint": bundle.fingerprint,
        }
    )
