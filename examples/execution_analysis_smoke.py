"""Small real-data capture check; never substitutes CPU for requested CUDA."""

import argparse
import json
from pathlib import Path

import torch

from efficientgraphbench import Benchmark, FidelityStage, ProgressiveBenchmark, ProgressiveConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="experiments/execution-review/texas")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; this smoke test will not silently switch to CPU")
    output = Path(args.output)
    profiling = {
        "level": "detailed",
        "epochs": [2],
        "memory_sampling": {"interval_ms": 20, "max_samples": 50},
    }
    full = Benchmark(
        dataset="texas",
        models=["gcn"],
        seeds=[1, 2],
        epochs=3,
        device=args.device,
        latency_warmup=1,
        latency_repeats=2,
        profiling=profiling,
        output_dir=str(output / "full"),
    ).run()
    progressive = ProgressiveBenchmark(
        dataset="texas",
        models=["gcn"],
        seeds=[1],
        device=args.device,
        output_dir=str(output / "progressive"),
        progressive=ProgressiveConfig(
            stages=[
                FidelityStage("initial", 0.5, 3),
                FidelityStage("full", 1.0, 3),
            ]
        ),
        profiling=profiling,
    ).run()
    evidence = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "profiling": profiling,
        "runs": [],
        "reports": {},
    }
    for mode, results in [("full", full), ("progressive", progressive)]:
        artifacts = results.execution_analysis()
        evidence["reports"][mode] = {key: str(value) for key, value in artifacts.items()}
        for row in results.records:
            execution = row.get("execution", {})
            evidence["runs"].append(
                {
                    "mode": mode,
                    "status": row["status"],
                    "failure_reason": row.get("failure_reason"),
                    "seed": row.get("seed"),
                    "stage_index": row.get("stage_index"),
                    "parent_checkpoint_id": row.get("parent_checkpoint_id"),
                    "checkpoint_id": row.get("checkpoint_id"),
                    "memory_sample_count": len(execution.get("memory_samples", [])),
                    "traces": execution.get("traces", []),
                    "operations": [
                        {
                            k: e.get(k)
                            for k in ["phase", "duration_ns", "cuda_elapsed_ms", "timing_scope"]
                        }
                        for e in execution.get("events", [])
                        if e["phase"]
                        in {
                            "training_forward",
                            "loss_calculation",
                            "training_backward",
                            "optimizer_step",
                            "checkpoint_restore_seconds",
                        }
                    ],
                    "profile_attempts": execution.get("profile_attempts", []),
                }
            )
    path = output / "evidence.json"
    path.write_text(json.dumps(evidence, indent=2), encoding="utf8")
    print(path)
    for run in evidence["runs"]:
        print(
            run["mode"],
            run["seed"],
            run["stage_index"],
            run["status"],
            "kernels:",
            sum(t["kernel_events"] for t in run["traces"]),
        )
    if any(run["status"] != "SUCCESS" for run in evidence["runs"]):
        raise SystemExit("Smoke run failed; inspect saved evidence")
    if args.device == "cuda" and not any(
        t["kernel_events"] for run in evidence["runs"] for t in run["traces"]
    ):
        raise SystemExit("CUDA training completed, but kernel capture was NOT verified")


if __name__ == "__main__":
    main()
