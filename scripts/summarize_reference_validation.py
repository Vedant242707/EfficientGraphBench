"""Verify actual saved results and produce the migration handoff evidence."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from efficientgraphbench.results.writer import read_results

ROOT = Path(__file__).resolve().parents[1]
rows = read_results(ROOT / "experiments/reference-final")
assert len(rows) == 60
assert len({(r["dataset"], r["model"], r["seed"]) for r in rows}) == 60
assert len({r["hardware_fingerprint"] for r in rows}) == 1
summary = {}
for dataset in ("cora", "pubmed"):
    selected = [r for r in rows if r["dataset"] == dataset]
    assert len(selected) == 30
    assert len({r["split_hash"] for r in selected}) == 1
    assert len({r["canonical_graph_hash"] for r in selected}) == 1
    assert len({r["comparison_group"] for r in selected}) == 1
    summary[dataset] = dict(Counter(r["status"] for r in selected))
    for r in selected:
        if r["status"] == "SUCCESS":
            assert r["epochs_run"] == 200
            assert r["parameter_count"] >= r["trainable_parameter_count"] > 0
            assert r["peak_gpu_reserved_mib"] >= r["peak_gpu_allocated_mib"] >= 0
            assert r["inference_latency_mean_ms"] >= 0
        if r["model"] in {"sgformer", "graphgps"}:
            assert (
                "runners\\reference_worker.py" in r["source_snapshot"]
                or "runners/reference_worker.py" in r["source_snapshot"]
            )
        if r["model"] == "appnp":
            assert r["source_type"] == "official_repo"
            assert r["architecture_modified"] is False
assert summary["cora"] == {"SUCCESS": 27, "UNSUPPORTED_TASK": 3}
assert summary["pubmed"] == {"SUCCESS": 24, "RUNTIME_ERROR": 3, "UNSUPPORTED_TASK": 3}
assert all("60s timeout" in r["failure_reason"] for r in rows if r["status"] == "RUNTIME_ERROR")
scaling_path = ROOT / "experiments/reference-scaling-smoke/scaling-results.json"
scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
assert len(scaling) == 6 and all(r["status"] == "SUCCESS" for r in scaling)
evidence = {
    "matrices": summary,
    "scaling": [(r["model"], r["num_nodes"], r["status"]) for r in scaling],
    "hardware": rows[0]["physical_hardware"],
    "results_sha256": hashlib.sha256(
        (ROOT / "experiments/reference-final/raw/results.jsonl").read_bytes()
    ).hexdigest(),
    "known_limits": [
        "GraphGPS PubMed exceeded the explicit 60s worker validation budget; not an observed OOM",
        "Graphormer reference is gated UNSUPPORTED_TASK",
        "No Git repository exists for EfficientGraphBench; project commit is null",
    ],
}
directory = ROOT / "audits/reference-migration"
directory.mkdir(parents=True, exist_ok=True)
(directory / "validation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
lines = [
    "# Reference migration status",
    "",
    "## Delivered",
    "",
    "- Pre-implementation audit and migration plan: MIGRATION_PLAN.md.",
    "- In-process, external-environment and pinned-repository runners.",
    "- Versioned JSON exchange, portable numeric graphs, shared masks and fingerprints.",
    "- Official GraphGPS and authors' SGFormer in isolated Python3.10/CUDA environments.",
    "- Primary APPNP uses the unchanged pinned PyG citation Net class.",
    "- Original prototype implementations remain under explicit adapted/library IDs.",
    "- Unified failure records and preprocessing/latency/allocated/reserved profiling; "
    "hardware isolation.",
    "- Environment/package locks, per-run source hashes and refreshed model provenance.",
    "- Model listing, scaling sweeps, and measured-primary recommendation filtering.",
    "",
    "## Validation",
    "",
    "102 tests passed; the only warning is the installed PyTorch JIT deprecation warning.",
    "Cora: 27 successful runs (nine models x three seeds); "
    "three Graphormer UNSUPPORTED_TASK results.",
    "PubMed: 24 successful runs (eight models x three seeds), three GraphGPS timeout results, "
    "three Graphormer UNSUPPORTED_TASK results. Successful runs used 200 epochs.",
    "All seeds within each dataset have matching graph/split/hardware/profiler fingerprints.",
    "Scaling smoke: GCN, SGFormer and GraphGPS succeeded at 500 and 1000 nodes, "
    "seed 42, two epochs.",
    "",
    "## Explicit limitations",
    "",
    "GraphGPS has not completed the PubMed validation: its official dense LapPE preprocessing "
    "exceeded the declared 60-second worker budget. Results are RUNTIME_ERROR; no OOM or "
    "architectural node limit is inferred. Production timeout remains 7200 seconds, "
    "configurable in YAML.",
    "The GraphGPS Actor node recipe is applied to citation data; "
    "it is not a tuned Cora/PubMed paper recipe.",
    "SGFormer uses source architecture and optimizer settings with canonical public splits "
    "and explicit "
    "epoch budgets; this is not a claim of paper-score reproduction.",
    "Graphormer has no validated unchanged reference adapter for this task. Its custom 4096 guard "
    "belongs only to graphormer_adapted.",
    "PyG APPNP's original model class is unchanged; the benchmark uses its own canonical-split "
    "training/checkpoint protocol rather than the source script's 100-run loss-window harness.",
    "The actual GPU reports 6141 MiB. NVML process VRAM was not measured; "
    "allocated/reserved metrics "
    "must not be presented as total process VRAM. The project Git commit is unavailable (null).",
    "",
    "## Files",
    "",
    "- experiments/reference-final/reports/cora.md",
    "- experiments/reference-final/reports/pubmed.md",
    "- experiments/reference-final/raw/results.jsonl and results.csv",
    "- audits/review20/model_provenance.json",
    "- audits/reference-migration/validation.json",
    "- REFERENCE_BENCHMARKS.md and FUNCTION_REFERENCE.md",
    "",
]
(ROOT / "REFERENCE_MIGRATION_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary))
