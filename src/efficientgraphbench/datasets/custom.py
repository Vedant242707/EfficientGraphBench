"""Portable user data: two CSV files or a numeric NPZ archive."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_custom(path: str | Path):
    path = Path(path)
    if path.is_dir():
        return _load_csv(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("custom dataset must be a folder with nodes.csv/edges.csv or a .npz file")
    required = {"x", "edge_index", "y", "train_mask", "val_mask", "test_mask"}
    # No pickle/object arrays: uploaded data cannot execute Python during loading.
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"NPZ is missing: {', '.join(sorted(missing))}")
        values = {name: archive[name] for name in required}
        optional = {
            name: archive[name]
            for name in ("raw_x", "edge_attr", "edge_weight")
            if name in archive.files
        }
    if values["edge_index"].dtype.kind not in "iu" or values["y"].dtype.kind not in "iu":
        raise ValueError("NPZ edge_index and y must be integer arrays")
    for name in ("train_mask", "val_mask", "test_mask"):
        if values[name].dtype.kind != "b":
            raise ValueError(f"NPZ {name} must contain boolean values")
    if values["x"].dtype.kind not in "fiu":
        raise ValueError("NPZ x must contain numeric features")
    data = Data(
        x=torch.from_numpy(values["x"].astype(np.float32)),
        edge_index=torch.from_numpy(values["edge_index"].astype(np.int64)),
        y=torch.from_numpy(values["y"].astype(np.int64)),
        **{name: torch.from_numpy(values[name]) for name in required if name.endswith("mask")},
    )
    for name, value in optional.items():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"NPZ {name} must contain finite numeric values")
        expected = data.num_nodes if name == "raw_x" else data.num_edges
        if value.ndim == 0 or len(value) != expected:
            raise ValueError(f"NPZ {name} has an invalid leading dimension")
        if name == "raw_x" and value.shape != values["x"].shape:
            raise ValueError("NPZ raw_x must match x shape")
        setattr(data, name, torch.from_numpy(value.astype(np.float32)))
    labels = torch.unique(data.y[data.y >= 0])
    classes = labels.numel()
    if not torch.equal(labels, torch.arange(classes)):
        raise ValueError("NPZ class IDs must be contiguous integers starting at zero")
    return data, classes, {str(i): i for i in range(classes)}


def _load_csv(path: Path, nodes_path=None, edges_path=None):
    nodes = pd.read_csv(nodes_path or path / "nodes.csv", dtype=str, keep_default_na=False)
    edges = pd.read_csv(edges_path or path / "edges.csv", dtype=str, keep_default_na=False)
    if not {"node_id", "label", "split"} <= set(nodes.columns):
        raise ValueError("nodes.csv requires node_id, label, split, and feature columns")
    if not {"source", "target"} <= set(edges.columns):
        raise ValueError(
            f"Missing source/target columns in {edges_path or path / 'edges.csv'}. Detected columns: {list(edges.columns)}. Rename endpoints to source/target or prepare with --source COLUMN --target COLUMN."  # noqa: E501
        )
    if nodes.node_id.eq("").any() or nodes.node_id.duplicated().any():
        raise ValueError("node_id values must be nonempty and unique")
    features = [column for column in nodes if column.startswith("feature_")]
    if not features:
        raise ValueError("nodes.csv needs numeric columns named feature_1, feature_2, etc.")
    try:
        x = torch.tensor(nodes[features].to_numpy(dtype=np.float32))
    except ValueError as exc:
        raise ValueError("feature columns must contain numbers, without missing values") from exc
    node_map = {identifier: index for index, identifier in enumerate(nodes.node_id)}
    endpoints = edges[["source", "target"]].apply(lambda column: column.map(node_map))
    if endpoints.isna().any().any():
        raise ValueError("edges.csv references node IDs missing from nodes.csv")
    labels = sorted(set(nodes.label) - {""})
    mapping = {label: index for index, label in enumerate(labels)}
    y = torch.tensor([mapping.get(label, -1) for label in nodes.label], dtype=torch.long)
    split = nodes.split.str.strip().str.lower().replace({"validation": "val"})
    if not split.isin(["train", "val", "test", "unused"]).all():
        raise ValueError("split must be train, val, test, or unused")
    data = Data(
        x=x,
        y=y,
        edge_index=torch.tensor(endpoints.to_numpy(dtype=np.int64).T),
        **{
            f"{name}_mask": torch.tensor((split == name).to_numpy())
            for name in ("train", "val", "test")
        },
    )
    attributes = [column for column in edges if column not in {"source", "target"}]
    if attributes:
        try:
            values = edges[attributes].to_numpy(dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                "Optional edge attributes must be numeric; encode categorical values explicitly"
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError("Edge attributes must be finite numbers")
        data.edge_attr = torch.from_numpy(values)
    return data, len(labels), mapping
