"""Disjoint synchronized phase timers; wall-clock totals are measured separately."""

import time
from contextlib import contextmanager

from efficientgraphbench.profiling.events import span


@contextmanager
def timed(totals: dict, key: str, device):
    with span(
        key,
        device=str(device),
        synchronization="phase-boundary CUDA sync" if device.type == "cuda" else "CPU completion",
    ):
        with _timed(totals, key, device):
            yield


@contextmanager
def _timed(totals: dict, key: str, device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            totals[key] = totals.get(key, 0.0) + time.perf_counter() - start
