"""Portable, pickle-free graph and canonical split exchange."""

import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_graph(bundle, directory, *, cache_dir=None):
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
    content_hash = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        graph_hash.update(key.encode())
        graph_hash.update(str(value.dtype).encode())
        graph_hash.update(str(value.shape).encode())
        if value.size:
            graph_hash.update(memoryview(np.ascontiguousarray(value)).cast("B"))
        if not key.endswith("_mask"):
            content_hash.update(key.encode())
            content_hash.update(str(value.dtype).encode())
            content_hash.update(str(value.shape).encode())
            if value.size:
                content_hash.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    splits = {
        key: np.flatnonzero(arrays[f"{key}_mask"]).tolist() for key in ("train", "val", "test")
    }
    split_hash = fingerprint(splits)
    graph_path = directory / "graph.npz"
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_path = cache_dir / f"{graph_hash.hexdigest()}.npz"
    checksum_path = graph_path.with_suffix(".sha256")
    cached_checksum = file_sha256(graph_path) if graph_path.exists() else None
    reusable = (
        cache_dir is not None
        and cached_checksum is not None
        and checksum_path.exists()
        and checksum_path.read_text(encoding="ascii") == cached_checksum
    )
    if not reusable:
        temporary = graph_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(graph_path)
        cached_checksum = file_sha256(graph_path)
        if cache_dir is not None:
            checksum_path.write_text(cached_checksum, encoding="ascii")
    split_path = directory / "split.json"
    split_path.write_text(json.dumps({**splits, "split_hash": split_hash}), encoding="utf-8")
    return {
        "graph_path": str(graph_path.resolve()),
        "split_path": str(split_path.resolve()),
        "split_hash": split_hash,
        "canonical_graph_hash": graph_hash.hexdigest(),
        "graph_content_hash": content_hash.hexdigest(),
        "graph_file_sha256": cached_checksum,
        "num_classes": bundle.num_classes,
        **bundle.metadata(),
    }
