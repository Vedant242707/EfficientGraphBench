"""Portable, pickle-free graph and canonical split exchange."""

import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def export_graph(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = bundle.data
    arrays = {
        key: getattr(data, key).detach().cpu().numpy()
        for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")
    }
    for key in ("edge_attr", "edge_weight", "raw_x"):
        if getattr(data, key, None) is not None:
            arrays[key] = getattr(data, key).detach().cpu().numpy()
    graph_hash = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        graph_hash.update(key.encode())
        graph_hash.update(str(value.dtype).encode())
        graph_hash.update(str(value.shape).encode())
        graph_hash.update(value.tobytes())
    splits = {
        key: np.flatnonzero(arrays[f"{key}_mask"]).tolist() for key in ("train", "val", "test")
    }
    split_hash = fingerprint(splits)
    graph_path = directory / "graph.npz"
    np.savez(graph_path, **arrays)
    split_path = directory / "split.json"
    split_path.write_text(json.dumps({**splits, "split_hash": split_hash}), encoding="utf-8")
    return {
        "graph_path": str(graph_path.resolve()),
        "split_path": str(split_path.resolve()),
        "split_hash": split_hash,
        "canonical_graph_hash": graph_hash.hexdigest(),
        "graph_file_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "num_classes": bundle.num_classes,
        **bundle.metadata(),
    }
