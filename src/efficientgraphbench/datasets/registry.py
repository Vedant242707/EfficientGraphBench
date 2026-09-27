from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikiCS,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

from efficientgraphbench.datasets.catalog import DATASETS, canonical_dataset
from efficientgraphbench.datasets.custom import load_custom
from efficientgraphbench.datasets.download import ReliableFlickr as Flickr
from efficientgraphbench.datasets.validation import (
    graph_fingerprint,
    stratified_masks,
    validate_graph,
)


@dataclass
class DatasetBundle:
    data: Data
    num_classes: int
    split: str = "planetoid-public"
    task_type: str = "node-classification"
    preprocessing: str = "row-normalized-features"
    edge_policy: str = "as-provided"
    fingerprint: str = ""
    label_mapping: dict = field(default_factory=dict)

    def metadata(self):
        data = self.data
        n = data.num_nodes
        return {
            "num_nodes": n,
            "num_edges": data.num_edges,
            "num_features": data.num_node_features,
            "num_classes": self.num_classes,
            "density": data.num_edges / (n * (n - 1)) if n > 1 else 0.0,
            "split": self.split,
            "task_type": self.task_type,
            "evaluation_protocol": "transductive-full-graph",
            "feature_preprocessing": self.preprocessing,
            "edge_policy": self.edge_policy,
            "dataset_fingerprint": self.fingerprint,
            "label_mapping": self.label_mapping,
            "train_nodes": int(data.train_mask.sum()),
            "val_nodes": int(data.val_mask.sum()),
            "test_nodes": int(data.test_mask.sum()),
            "training_class_counts": torch.bincount(
                data.y[data.train_mask], minlength=self.num_classes
            ).tolist(),
            "missing_training_classes": sorted(
                set(range(self.num_classes)) - set(data.y[data.train_mask].tolist())
            ),
        }


def load_dataset(
    name: str,
    root: str | Path,
    *,
    dataset_path: str | None = None,
    split_seed: int = 0,
    split_index: int = 0,
    split_mode: str = "official",
) -> DatasetBundle:
    if split_mode not in {"official", "random"}:
        raise ValueError("split_mode must be official or random")
    name = canonical_dataset(name)
    if name not in DATASETS:
        raise ValueError(f"unsupported dataset: {name}; use egbench datasets to see choices")
    spec = DATASETS[name]
    directory = str(Path(root) / name)
    preprocessing, edge_policy, mapping = "as-provided", "as-provided", {}
    if spec.loader == "custom":
        if not dataset_path:
            raise ValueError("custom data requires --dataset-path (CSV folder or NPZ file)")
        data, classes, mapping = load_custom(dataset_path)
        split = "user-provided"
    elif spec.loader == "ogb":
        data, classes = _load_ogb(name, root)
        split = "ogb-official-time" if name == "ogbn-arxiv" else "ogb-official-sales-rank"
        edge_policy = (
            "converted-to-undirected" if name == "ogbn-arxiv" else "as-provided-undirected"
        )
    else:
        if spec.loader == "planetoid":
            dataset = Planetoid(directory, spec.argument, split="public")
        elif spec.loader == "amazon":
            dataset = Amazon(directory, spec.argument)
        elif spec.loader == "coauthor":
            dataset = Coauthor(directory, spec.argument)
        elif spec.loader == "wikics":
            dataset = WikiCS(directory, is_undirected=True)
            edge_policy = "converted-to-undirected"
        elif spec.loader == "flickr":
            dataset = Flickr(directory)
        elif spec.loader == "heterophilous":
            dataset = HeterophilousGraphDataset(directory, spec.argument)
        elif spec.loader == "actor":
            dataset = Actor(directory)
        elif spec.loader == "webkb":
            dataset = WebKB(directory, spec.argument)
        else:
            raise ValueError(f"unrecognized dataset loader: {spec.loader}")
        data, classes = dataset[0], dataset.num_classes
        if spec.loader in {"planetoid", "amazon", "coauthor"}:
            data.raw_x = data.x.clone()
            data = NormalizeFeatures()(data)
            preprocessing = "pyg-normalize-features"
        if spec.split == "generated":
            data.train_mask, data.val_mask, data.test_mask = stratified_masks(data.y, split_seed)
            split = f"stratified-per-class-60-20-20-seed{split_seed}"
        elif data.train_mask.ndim == 2:
            if not 0 <= split_index < data.train_mask.shape[1]:
                raise ValueError(
                    f"{name} split_index must be between 0 and {data.train_mask.shape[1] - 1}"
                )
            for key in ("train_mask", "val_mask", "test_mask"):
                mask = getattr(data, key)
                if mask.ndim == 2:
                    setattr(data, key, mask[:, split_index])
            split = f"{name}-official-{split_index}"
        else:
            split = "planetoid-public" if spec.loader == "planetoid" else "flickr-official"
    if split_mode == "random":
        data.train_mask, data.val_mask, data.test_mask = stratified_masks(data.y, split_seed)
        split = f"nonofficial-stratified-per-class-60-20-20-seed{split_seed}"
    validate_graph(data, classes)
    return DatasetBundle(
        data,
        classes,
        split=split,
        preprocessing=preprocessing,
        edge_policy=edge_policy,
        fingerprint=graph_fingerprint(data),
        label_mapping=mapping,
    )


def _load_ogb(name, root):
    try:
        # NumPy loader avoids loading pickled PyG objects from third-party caches.
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ValueError("This dataset requires OGB: python -m pip install 'ogb>=1.3.6'") from exc
    directory = Path(root) / name.replace("-", "_")
    cache = directory / "processed" / "egbench-numeric-v1.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            graph = {key: stored[key] for key in ("node_feat", "edge_index")}
            labels = stored["labels"]
            classes = int(stored["num_classes"])
            splits = {key: stored[key] for key in ("train", "valid", "test")}
    else:
        raw = directory / "raw"
        if (raw / "edge.csv.gz").exists():
            # OGB's existing protocol-4 pickle cache is incompatible with
            # newer torch.load defaults. Rebuild from official numeric CSVs.
            import pandas as pd
            from ogb.io.read_graph_raw import read_csv_graph_raw

            graph = read_csv_graph_raw(str(raw), add_inverse_edge=name == "ogbn-products")[0]
            labels = pd.read_csv(raw / "node-label.csv.gz", header=None).to_numpy(copy=True)
            classes = 40 if name == "ogbn-arxiv" else 47
            split_dir = directory / "split" / ("time" if name == "ogbn-arxiv" else "sales_ranking")
            splits = {
                key: pd.read_csv(split_dir / f"{key}.csv.gz", header=None)
                .to_numpy(copy=True)
                .reshape(-1)
                for key in ("train", "valid", "test")
            }
        else:
            dataset = NodePropPredDataset(name=name, root=str(root))
            graph, labels = dataset[0]
            classes = dataset.num_classes
            splits = dataset.get_idx_split()
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                node_feat=graph["node_feat"],
                edge_index=graph["edge_index"],
                labels=labels,
                num_classes=classes,
                **splits,
            )
        temporary.replace(cache)
    data = Data(
        x=torch.as_tensor(graph["node_feat"], dtype=torch.float32),
        edge_index=(
            to_undirected(torch.as_tensor(graph["edge_index"], dtype=torch.long))
            if name == "ogbn-arxiv"
            else torch.as_tensor(graph["edge_index"], dtype=torch.long)
        ),
        y=torch.tensor(labels, dtype=torch.long).reshape(-1),
    )
    for source, target in (("train", "train_mask"), ("valid", "val_mask"), ("test", "test_mask")):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(splits[source], dtype=torch.long)] = True
        setattr(data, target, mask)
    return data, classes
