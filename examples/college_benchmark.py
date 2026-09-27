"""Run from the college SSH terminal: python examples/college_benchmark.py."""

from pathlib import Path

import torch

from efficientgraphbench import Benchmark

# Edit these four settings. Dataset download starts when run() is called.
DATASET = "ogbn-arxiv"
MODELS = ["gcn", "graphsage", "gat", "gatv2", "appnp", "sgc", "mlp"]
RUNS = 10
SPLITS = "fixed"  # fixed: training seeds only; official: cycle folds; random: new 60/20/20 splits


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment. Check the college venv.")
    print("GPU:", torch.cuda.get_device_name(0))
    output = Path("experiments") / f"college-{DATASET}-{SPLITS}-{RUNS}"
    results = Benchmark(
        dataset=DATASET,
        models=MODELS,
        runs=RUNS,
        splits=SPLITS,
        epochs=200,
        device="cuda",
        output_dir=str(output),
    ).run()
    summary = results.summary()
    print(
        summary[
            ["model", "runs", "distinct_splits", "accuracy_mean", "accuracy_sd", "status"]
        ].to_string(index=False)
    )
    summary.to_csv(output / "summary.csv", index=False)
    results.save_json(output / "runs.json")
    results.save_csv(output / "runs.csv")
    print("Results saved in:", output.resolve())


if __name__ == "__main__":
    main()
