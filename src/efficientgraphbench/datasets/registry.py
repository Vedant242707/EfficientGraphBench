from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, WikiCS
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
        }


def load_dataset(
    name: str,
    root: str | Path,
    *,
    dataset_path: str | None = None,
    split_seed: int = 0,
    split_index: int = 0,
) -> DatasetBundle:
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
        split = "ogb-official-time"
        edge_policy = "converted-to-undirected"
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
        elif spec.loader == "wikics":
            if not 0 <= split_index < data.train_mask.shape[1]:
                raise ValueError("WikiCS split_index must be between 0 and 19")
            data.train_mask = data.train_mask[:, split_index]
            data.val_mask = data.val_mask[:, split_index]
            split = f"wikics-official-{split_index}"
        else:
            split = "planetoid-public" if spec.loader == "planetoid" else "flickr-official"
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
        raise ValueError(
            "ogbn-arxiv requires OGB: uv pip install --python .venv/Scripts/python.exe -e '.[ogb]'"
        ) from exc
    dataset = NodePropPredDataset(name=name, root=str(root))
    graph, labels = dataset[0]
    data = Data(
        x=torch.as_tensor(graph["node_feat"], dtype=torch.float32),
        edge_index=to_undirected(torch.as_tensor(graph["edge_index"], dtype=torch.long)),
        y=torch.as_tensor(labels, dtype=torch.long).reshape(-1),
    )
    splits = dataset.get_idx_split()
    for source, target in (("train", "train_mask"), ("valid", "val_mask"), ("test", "test_mask")):
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(splits[source], dtype=torch.long)] = True
        setattr(data, target, mask)
    return data, dataset.num_classes
