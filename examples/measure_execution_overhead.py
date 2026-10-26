"""Small controlled instrumentation experiment; never a model-ranking benchmark."""

import json
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from efficientgraphbench import Benchmark
from efficientgraphbench.datasets.registry import DatasetBundle


def main():
    gen = torch.Generator().manual_seed(51)
    ids = torch.arange(500)
    bundle = DatasetBundle(
        Data(
            x=torch.randn(500, 16, generator=gen),
            y=ids % 3,
            edge_index=torch.randint(500, (2, 2000), generator=gen),
            train_mask=ids % 5 < 3,
            val_mask=ids % 5 == 3,
            test_mask=ids % 5 == 4,
        ),
        3,
    )
    destination = Path("experiments/execution-overhead")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    root = Path(tempfile.mkdtemp(prefix="egbench-overhead-"))
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        # Discard a warm-up run. Then rotate order to reduce fixed ordering effects.
        Benchmark(
            bundle,
            models=["gcn"],
            seeds=[7],
            device=device,
            epochs=2,
            profiling={"level": "none"},
            output_dir=str(root / (device + "-warmup")),
        ).run()
        for repeat in range(3):
            levels = ["none", "lightweight", "detailed"]
            levels = levels[repeat:] + levels[:repeat]
            for level in levels:
                started = time.perf_counter()
                row = (
                    Benchmark(
                        bundle,
                        models=["gcn"],
                        seeds=[7],
                        device=device,
                        epochs=50,
                        latency_warmup=2,
                        latency_repeats=5,
                        profiling={"level": level, "epochs": [10]},
                        output_dir=str(root / f"{device}-{repeat}-{level}"),
                    )
                    .run()
                    .records[0]
                )
                if row["status"] != "SUCCESS":
                    raise RuntimeError(row.get("failure_reason"))
                rows.append(
                    dict(
                        device=device,
                        level=level,
                        repeat=repeat,
                        wall_seconds=time.perf_counter() - started,
                        training_seconds=row["training_time_sec"],
                        gpu_allocated_mib=row.get("peak_gpu_allocated_mib"),
                        cpu_rss_mib=row.get("peak_cpu_memory_mb"),
                        accuracy=row["accuracy"],
                    )
                )
    summaries = []
    for device in sorted({r["device"] for r in rows}):
        for level in ["none", "lightweight", "detailed"]:
            subset = [r for r in rows if r["device"] == device and r["level"] == level]
            baseline = [r for r in rows if r["device"] == device and r["level"] == "none"]
            item = dict(device=device, level=level)
            for metric in ["wall_seconds", "training_seconds", "gpu_allocated_mib", "cpu_rss_mib"]:
                values = [r[metric] for r in subset if r[metric] is not None]
                reference = [r[metric] for r in baseline if r[metric] is not None]
                item[metric] = statistics.median(values) if values else None
                item[metric + "_overhead_percent"] = (
                    100 * (statistics.median(values) / statistics.median(reference) - 1)
                    if values and reference and statistics.median(reference) > 0
                    else None
                )
            summaries.append(item)
    result = dict(
        scope="GCN; 500 nodes; 2000 edges; 16 features; 50 epochs; seed7; "
        "3 order-rotated repeats after warmup; detailed epoch10 only. "
        "Same-process RSS is not incremental profiler memory. "
        "Small workload overhead is not a large-graph prediction.",
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        raw=rows,
        medians=summaries,
        run_artifacts=str(root),
    )
    path = destination / "overhead.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(path.resolve())


if __name__ == "__main__":
    main()
