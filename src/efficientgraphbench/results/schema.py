"""Additive schema normalization; historical records keep their original schema."""


def normalize_record(record):
    if record.get("schema_version", 0) < 4:
        return record
    from efficientgraphbench.version import __version__

    record.setdefault("efficientgraphbench_version", __version__)
    record.setdefault("failure_stage", None)
    record.setdefault("model_id", record.get("model"))
    record.setdefault("display_name", record.get("model_display_name", record.get("model")))
    for key in (
        "repository_url",
        "repository_commit",
        "environment",
        "environment_specification",
        "implementation_type",
        "source_type",
        "software",
        "split_hash",
        "dataset_fingerprint",
        "hardware_fingerprint",
        "profiler_settings",
        "efficientgraphbench_commit",
        "parameter_count",
        "trainable_parameter_count",
        "preprocessing_time_sec",
        "training_time_sec",
        "inference_latency_mean_ms",
        "inference_latency_median_ms",
        "inference_latency_std_ms",
        "peak_gpu_allocated_mib",
        "peak_gpu_reserved_mib",
        "epochs_run",
        "early_stopping_patience",
        "training_termination_reason",
        "scheduler",
        "failure_reason",
        "implementation_limit",
    ):
        record.setdefault(key, None)
    config = record.get("config", {})
    record.setdefault("maximum_epochs", config.get("training", {}).get("epochs"))
    return record
