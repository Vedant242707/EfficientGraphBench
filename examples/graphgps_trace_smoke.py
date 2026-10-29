"""Check isolated GraphGPS capture identities and CUDA capability without model substitutions."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from efficientgraphbench import Benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/graphgps-trace-audit")
    parser.add_argument("--require-cuda-kernels", action="store_true")
    args = parser.parse_args()
    results = Benchmark(
        dataset="texas",
        models=["graphgps"],
        seeds=[1, 2],
        epochs=2,
        device="cuda",
        latency_warmup=1,
        latency_repeats=2,
        output_dir=args.output,
        profiling={"level": "detailed", "epochs": [1, 2]},
    ).run()
    report = results.execution_analysis()
    payload = json.loads(report["data"].read_text(encoding="utf8"))
    evidence = {"bundle": str(report["bundle"]), "runs": [], "manifest": payload["trace_manifest"]}
    for row in results.records:
        execution = row.get("worker_execution", {})
        traces = execution.get("traces", [])
        evidence["runs"].append(
            {
                "seed": row["seed"],
                "status": row["status"],
                "failure_reason": row.get("failure_reason"),
                "diagnostics": execution.get("profile_attempts", []),
                "traces": traces,
                "cuda_operation_intervals_ms": [
                    e["cuda_elapsed_ms"]
                    for e in execution.get("events", [])
                    if e.get("cuda_elapsed_ms") is not None
                ],
            }
        )
    output = Path(args.output) / "trace-evidence.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf8")
    assert all(r["status"] == "SUCCESS" for r in evidence["runs"]), str(output)
    refs = [t for s in payload["sessions"] for t in s.get("traces", [])]
    assert len(refs) == len({t["path"] for t in refs}) == 4
    with zipfile.ZipFile(report["bundle"]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".trace.json")]
        assert len(names) == len(set(names)) == len(refs)
        for trace in refs:
            raw = archive.read(trace["path"])
            assert hashlib.sha256(raw).hexdigest() == trace["sha256"]
            captured = json.loads(raw)
            assert captured["efficientgraphbench"]["capture_id"] == trace["capture_id"]
            assert captured["efficientgraphbench"]["identity"]["seed"] == trace["identity"]["seed"]
    kernels = sum(t["kernel_events"] for t in refs)
    print("Four independently preserved captures; all ZIP paths and checksums verified.")
    print("Measured CUDA kernel events:", kernels)
    print("Evidence:", output)
    if not kernels:
        print(
            "CUDA kernel capture NOT verified in this isolated environment. "
            "See diagnostics; no device or model fallback was applied."
        )
    if args.require_cuda_kernels:
        assert all(t["kernel_events"] > 0 for t in refs), (
            "GraphGPS CUDA kernel capture not verified for every requested capture; " + str(output)
        )


if __name__ == "__main__":
    main()
