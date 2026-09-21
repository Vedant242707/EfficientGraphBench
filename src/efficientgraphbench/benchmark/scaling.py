"""Deterministic induced-subgraph scaling, without relabeling the task or regenerating splits."""

import copy
import json
import uuid
from pathlib import Path

import numpy as np
import torch

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.datasets.registry import load_dataset
from efficientgraphbench.results.status import status_of


def run_scaling(config, sizes, models, seeds, step=None):
    config.validate()
    bundle = load_dataset(
        config.dataset,
        config.data_dir,
        dataset_path=config.dataset_path,
        split_seed=config.split_seed,
        split_index=config.split_index,
    )
    data = bundle.data
    if step is not None:
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive node count")
        sizes = list(range(step, data.num_nodes, step)) + [data.num_nodes]
    sizes = sorted(set(sizes))
    models, seeds = list(dict.fromkeys(models)), list(dict.fromkeys(seeds))
    if not sizes or not models or not seeds:
        raise ValueError("Sizes, models and seeds must not be empty")
    generator = torch.Generator().manual_seed(config.split_seed)
    # Retain one labeled node per class per available split, then add a fixed random order.
    required = set()
    for split in ("train", "val", "test"):
        for label in torch.unique(data.y[data.y >= 0]):
            matches = ((data.y == label) & getattr(data, f"{split}_mask")).nonzero().flatten()
            if len(matches):
                required.add(int(matches[0]))
    order = sorted(required) + [
        int(n)
        for n in torch.randperm(data.num_nodes, generator=generator)
        if int(n) not in required
    ]
    if any(size < len(required) or size > data.num_nodes for size in sizes):
        raise ValueError(f"Sizes must be in [{len(required)}, {data.num_nodes}] for this dataset")
    outputs = []
    stopped = set()
    outcomes = {model: {"last_success": None, "first_oom": None} for model in models}
    skipped = []
    summary = Path(config.output_dir) / "scaling-results.json"
    for size in sizes:
        root = Path(config.output_dir) / f"nodes-{size}"
        root.mkdir(parents=True, exist_ok=True)
        selected = torch.tensor(sorted(order[:size]))
        graph = data.subgraph(selected)
        path = root / "graph.npz"
        np.savez(
            path,
            **{
                key: getattr(graph, key).cpu().numpy()
                for key in (
                    "x",
                    "edge_index",
                    "y",
                    "train_mask",
                    "val_mask",
                    "test_mask",
                    "raw_x",
                    "edge_attr",
                    "edge_weight",
                )
                if getattr(graph, key, None) is not None
            },
        )
        context = {
            "parent_dataset": config.dataset,
            "parent_fingerprint": bundle.metadata()["dataset_fingerprint"],
            "method": "deterministic induced subgraph; original split memberships retained",
            "original_node_indices": selected.tolist(),
            "node_count": size,
            "edge_count": graph.num_edges,
            "sampling_seed": config.split_seed,
            "accuracy_note": "Accuracy across sizes is not the same prediction problem",
        }
        (root / "scaling.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
        batch = uuid.uuid4().hex
        deterministic_failures = {}
        for model in models:
            for seed in seeds:
                if model in stopped or model in deterministic_failures:
                    skipped.append(
                        {
                            "model": model,
                            "seed": seed,
                            "num_nodes": size,
                            "status": "SKIPPED_AFTER_OOM"
                            if model in stopped
                            else "SKIPPED_DETERMINISTIC_FAILURE",
                            "reason": "Skipped after observed failure; not measured",
                        }
                    )
                    continue
                job = copy.deepcopy(config)
                job.dataset, job.dataset_path = "custom", str(path.resolve())
                job.output_dir, job.seed, job.model.name = str(root), seed, model
                job.split_index = 0
                job._result_batch, job._scaling_context = batch, context
                result = run_benchmark(job)
                if result.get("failure_stage") in {
                    "configuration",
                    "dependencies",
                    "preprocessing",
                    "allocation",
                } and status_of(result) in {
                    "UNSUPPORTED_TASK",
                    "UNSUPPORTED_GRAPH_SIZE",
                    "CONFIG_ERROR",
                    "DEPENDENCY_ERROR",
                    "ENVIRONMENT_NOT_INSTALLED",
                    "PREPROCESSING_ERROR",
                }:
                    deterministic_failures[model] = result
                outputs.append(result)
                outcomes[model]["last_status"] = status_of(result)
                measured = {
                    "num_nodes": size,
                    "num_edges": graph.num_edges,
                    "seed": seed,
                    "result": result,
                }
                outcomes[model]["last_executed"] = measured
                if status_of(result) == "SUCCESS":
                    outcomes[model]["last_success"] = measured
                elif status_of(result) == "OOM":
                    outcomes[model]["first_oom"] = measured
                    stopped.add(model)
                # Save progress after every measured run, including failures.
                summary.write_text(json.dumps(outputs, indent=2, allow_nan=False), encoding="utf-8")
        (Path(config.output_dir) / "scaling-summary.json").write_text(
            json.dumps(
                {
                    "unit": "nodes",
                    "sizes": sizes,
                    "models": outcomes,
                    "skipped_runs": skipped,
                    "note": "Last success is an observed run, not a guaranteed capacity. "
                    "Each size trains a fresh model on an induced subgraph.",
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return summary
