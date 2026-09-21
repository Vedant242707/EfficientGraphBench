"""Refresh reference entries in the existing validation batches, preserving PyG measurements."""

import copy
from pathlib import Path

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.config import load_config
from efficientgraphbench.models.provenance import generate_provenance
from efficientgraphbench.results.report import write_report
from efficientgraphbench.results.writer import read_results

base = load_config(Path("configs/reference-validation.yaml"))
for dataset in ("cora", "pubmed"):
    rows = [r for r in read_results(base.output_dir) if r["dataset"] == dataset]
    batch = rows[-1]["batch_id"]
    assert batch, "Expected an existing validation matrix batch"
    for model in ("appnp", "sgformer", "graphgps"):
        for seed in (42, 43, 44):
            cfg = copy.deepcopy(base)
            cfg.dataset, cfg.model.name, cfg.seed = dataset, model, seed
            cfg._result_batch = batch
            result = run_benchmark(cfg)
            print(dataset, model, seed, result["status"], result.get("failure_reason"), flush=True)
    print(write_report(dataset, base.output_dir), flush=True)
print(generate_provenance(), flush=True)
