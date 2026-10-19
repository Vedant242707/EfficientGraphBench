"""Persistent stage exchanges through the existing pinned repository runner."""

import time
import uuid
from pathlib import Path

from efficientgraphbench.benchmark.execution import RepositoryRunner, RunnerError
from efficientgraphbench.models.catalog import model_spec


def train_reference_stage(
    config,
    progressive,
    stage,
    graph,
    index,
    classes,
    device,
    output,
    provenance,
    parent,
    hardware_hash,
):
    spec = model_spec(config.model.name)
    if not Path(spec["python"]).exists():
        raise RunnerError("ENVIRONMENT_NOT_INSTALLED", "Run egbench setup " + spec["environment"])
    request = {
        **graph.metadata,
        "run_id": uuid.uuid4().hex,
        "model_id": config.model.name,
        "seed": config.seed,
        "hardware_fingerprint": hardware_hash,
        "device": str(device),
        "config": config.to_dict(),
        "repository": spec["repository"],
        "num_classes": classes,
        "controller_preprocessing_time_sec": 0,
        "benchmark_mode": "progressive_multi_fidelity",
        "progressive": progressive.to_dict(),
        "stage_index": index,
        "old_node_count": int(graph.old_mask.sum()),
        "stage_metadata": graph.metadata,
        "parent": parent,
        "output_dir": str(output.resolve()),
        "controller_provenance": provenance,
    }
    worker = Path(__file__).resolve().parents[1] / "runners/reference_worker.py"
    runner = RepositoryRunner(
        [spec["python"], str(worker)],
        output / "exchange" / config.model.name / str(config.seed) / str(index),
        spec["repository"],
        spec["commit"],
        timeout=config.worker_timeout_sec,
    )
    start = time.perf_counter()
    result = runner.run(request)
    result["worker_wall_time_sec"] = time.perf_counter() - start
    return result
