"""Validate the node-classification contract before allocating model parameters."""

import hashlib

import torch


def validate_graph(data, num_classes: int):
    if data.x is None or data.x.ndim != 2 or min(data.x.shape) < 1:
        raise ValueError("features x must have shape [nodes, features] with nonzero dimensions")
    if not data.x.is_floating_point() or not bool(torch.isfinite(data.x).all()):
        raise ValueError("features must be finite floating-point numbers")
    n = data.x.shape[0]
    edges = data.edge_index
    if edges is None or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be an integer tensor with shape [2, edges]")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n):
        raise ValueError("edges reference a node outside the feature table")
    if data.y is None or data.y.dtype != torch.long or data.y.shape != (n,):
        raise ValueError("labels y must be a one-dimensional integer tensor, one per node")
    if num_classes < 2:
        raise ValueError("classification requires at least two classes")
    if bool(((data.y < -1) | (data.y >= num_classes)).any()):
        raise ValueError("labels must be -1 (unlabeled) or class IDs from 0 to num_classes - 1")
    occupied = torch.zeros(n, dtype=torch.bool)
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is None or mask.dtype != torch.bool or mask.shape != (n,):
            raise ValueError(f"{name} must be a boolean vector with one entry per node")
        if not bool(mask.any()):
            raise ValueError(f"{name} cannot be empty")
        if bool((occupied & mask).any()):
            raise ValueError("train, validation, and test splits must not overlap")
        if bool((data.y[mask] < 0).any()):
            raise ValueError(f"{name} includes unlabeled nodes")
        occupied |= mask
    if torch.unique(data.y[data.train_mask]).numel() != num_classes:
        raise ValueError("the training split must contain at least one example of every class")
    return data


def graph_fingerprint(data) -> str:
    """Hash content and splits, preventing unrelated custom graphs being averaged."""
    digest = hashlib.sha256()
    for name in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask"):
        array = getattr(data, name).detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def stratified_masks(labels, seed: int):
    """Per-class 60/20/20 split; model initialization seeds never change it."""
    generator = torch.Generator().manual_seed(seed)
    masks = [torch.zeros(labels.numel(), dtype=torch.bool) for _ in range(3)]
    for label in torch.unique(labels):
        nodes = (labels == label).nonzero(as_tuple=True)[0]
        if nodes.numel() < 3:
            raise ValueError("generated splits need at least three nodes per class")
        nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
        train_count = min(max(1, int(len(nodes) * 0.6)), len(nodes) - 2)
        val_count = min(max(1, int(len(nodes) * 0.2)), len(nodes) - train_count - 1)
        masks[0][nodes[:train_count]] = True
        masks[1][nodes[train_count : train_count + val_count]] = True
        masks[2][nodes[train_count + val_count :]] = True
    return masks
