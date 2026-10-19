"""Disjoint synchronized phase timers; wall-clock totals are measured separately."""

import time
from contextlib import contextmanager


@contextmanager
def timed(totals: dict, key: str, device):
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
