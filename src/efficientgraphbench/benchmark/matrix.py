"""Shared sequential matrix scheduling for Python and CLI."""

import copy
import uuid

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.results.status import status_of
from efficientgraphbench.results.writer import write_result


def run_matrix(jobs, execute=None):
    execute = execute or run_benchmark
    batch = uuid.uuid4().hex
    records, blocked = [], {}
    for config in jobs:
        config._result_batch = batch
        model = config.model.name
        if model in blocked:
            cause = blocked[model]
            deterministic = cause.get("failure_stage") in {
                "configuration",
                "dependencies",
                "preprocessing",
                "allocation",
            }
            result = copy.deepcopy(cause)
            for key in (
                "accuracy",
                "validation_accuracy",
                "training_time_sec",
                "preprocessing_time_sec",
                "inference_latency_mean_ms",
                "inference_latency_median_ms",
                "inference_latency_std_ms",
                "peak_gpu_allocated_mib",
                "peak_gpu_reserved_mib",
                "peak_cpu_memory_mb",
                "epochs_run",
                "inference_ms",
                "inference_mean_ms",
                "peak_gpu_memory_mb",
                "train_time_sec",
                "checkpoint_path",
                "worker_wall_time_sec",
                "elapsed_until_failure_sec",
            ):
                result.pop(key, None)
            result.update(
                run_id=uuid.uuid4().hex,
                seed=config.seed,
                config=config.to_dict(),
                status="SKIPPED_DETERMINISTIC_FAILURE" if deterministic else "SKIPPED_AFTER_OOM",
                failure_reason=f"Not executed after {cause['status']} in run {cause['run_id']}",
                skipped_after_run_id=cause["run_id"],
                measured=False,
            )
            write_result(result, config.output_dir)
        else:
            result = execute(config)
            result.setdefault("run_id", uuid.uuid4().hex)
            result.setdefault("model", model)
            result.setdefault("batch_id", batch)
            if status_of(result) == "OOM" or (
                result.get("failure_stage")
                in {"configuration", "dependencies", "preprocessing", "allocation"}
                and status_of(result)
                in {
                    "UNSUPPORTED_TASK",
                    "UNSUPPORTED_GRAPH_SIZE",
                    "CONFIG_ERROR",
                    "DEPENDENCY_ERROR",
                    "ENVIRONMENT_NOT_INSTALLED",
                    "PREPROCESSING_ERROR",
                }
            ):
                blocked[model] = result
        records.append(result)
    return records
