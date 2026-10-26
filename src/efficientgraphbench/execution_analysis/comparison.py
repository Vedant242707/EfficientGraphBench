"""Conservative CPU/GPU matching. Missing provenance means no speedup claim."""

import json
import statistics


def device_comparisons(records):
    groups = {}
    for row in records:
        if row.get("status") != "SUCCESS" or row.get("benchmark_mode", "full") != "full":
            continue
        definition = row.get("execution_identity", {}).get("definition")
        if not definition or row.get("epochs_run") is None:
            continue
        workload = {k: v for k, v in definition.items() if k != "hardware"}
        workload["epochs_run"] = row["epochs_run"]
        workload["scope"] = row.get("profiler_settings")
        key = json.dumps(workload, sort_keys=True)
        device = row.get("device", "").split(":")[0]
        groups.setdefault(key, {}).setdefault(device, []).append(row)
    comparisons = []
    for devices in groups.values():
        # Avoid selecting an arbitrary repeat or machine from several possible matches.
        if len(devices.get("cpu", [])) != 1 or len(devices.get("cuda", [])) != 1:
            continue
        cpu, gpu = devices["cpu"][0], devices["cuda"][0]
        if any(
            not isinstance(r.get("training_time_sec"), (int, float)) or r["training_time_sec"] <= 0
            for r in (cpu, gpu)
        ):
            continue

        def summary(row):
            epochs = row.get("epoch_times_sec", [])
            return dict(
                run_id=row["run_id"],
                training_seconds=row["training_time_sec"],
                median_epoch_seconds=statistics.median(epochs) if epochs else None,
                latency_ms=row.get("inference_latency_median_ms"),
                cpu_rss_mib=row.get("peak_cpu_memory_mb"),
                gpu_reserved_mib=row.get("peak_gpu_reserved_mib"),
                hardware=row.get("physical_hardware"),
            )

        a, b = cpu.get("physical_hardware", {}), gpu.get("physical_hardware", {})
        comparisons.append(
            dict(
                model=cpu["model"],
                seed=cpu["seed"],
                cpu=summary(cpu),
                cuda=summary(gpu),
                training_speedup=cpu["training_time_sec"] / gpu["training_time_sec"],
                instrumented=cpu.get("instrumented"),
                same_host=bool(a.get("machine"))
                and all(a.get(k) == b.get(k) for k in ("machine", "cpu", "ram_bytes")),
            )
        )
    return comparisons
