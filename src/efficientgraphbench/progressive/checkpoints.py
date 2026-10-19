"""Continuation checkpoints extend the existing model_state_dict checkpoint key."""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from efficientgraphbench.benchmark.interchange import file_sha256


class CheckpointManager:
    @staticmethod
    def signature(value):
        digest = hashlib.sha256()

        def visit(item):
            if torch.is_tensor(item):
                tensor = item.detach().cpu().contiguous()
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.numpy().tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    digest.update(str(key).encode())
                    visit(item[key])
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)
            else:
                digest.update(repr(item).encode())

        visit(value)
        return digest.hexdigest()

    @staticmethod
    def save(path: Path, model, optimizer, scheduler, metadata: dict) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": "egbench-progressive-v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            **metadata,
        }
        state["saved_model_signature"] = CheckpointManager.signature(model.state_dict())
        state["saved_optimizer_signature"] = CheckpointManager.signature(optimizer.state_dict())
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return file_sha256(path)

    @staticmethod
    def restore(path: Path, checksum: str, expected: dict, model, optimizer, scheduler) -> dict:
        if file_sha256(path) != checksum:
            raise ValueError("Checkpoint hash mismatch")
        # Only locally generated checkpoints with a controller-held digest are accepted.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format") != "egbench-progressive-v1":
            raise ValueError("Not a progressive continuation checkpoint")
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Checkpoint lineage mismatch: {key}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        for label, restored in (
            ("model", model.state_dict()),
            ("optimizer", optimizer.state_dict()),
        ):
            key = "saved_" + label + "_signature"
            if key in state and CheckpointManager.signature(restored) != state[key]:
                raise ValueError("Restored " + label + " differs from saved checkpoint")
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"])
        if state["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_state"])
        return state
