"""Use only explicitly supplied, compatible measured full-benchmark evidence."""

import json
import math
from pathlib import Path

from efficientgraphbench.benchmark.interchange import file_sha256


def finite_nonnegative(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def load_full_baseline(path: str | None, expected: dict) -> dict:
    unavailable = {
        "full_benchmark_baseline": "unavailable",
        "training_seconds": None,
        "elapsed_seconds": None,
        "reason": "No saved full baseline supplied",
    }
    if path is None:
        return unavailable
    try:
        source = Path(path)
        if source.is_dir():
            source = source / (
                "full-benchmark.json" if (source / "full-benchmark.json").exists() else "runs.json"
            )
        raw = (
            [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if source.suffix == ".jsonl"
            else json.loads(source.read_text(encoding="utf-8"))
        )
        metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        rows = raw.get("records", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list) or not rows:
            raise ValueError("Baseline must contain full run records")
        pairs = [(r.get("model"), r.get("seed")) for r in rows]
        required = {(m, s) for m in expected["models"] for s in expected["seeds"]}
        if set(pairs) != required or len(pairs) != len(required):
            raise ValueError("Candidate/seed coverage differs or duplicate attempts are present")
        for row in rows:
            if row.get("status") != "SUCCESS" or row.get("benchmark_mode", "full") != "full":
                raise ValueError("Baseline must contain successful static full runs only")
            for key in ("canonical_graph_hash", "split_hash", "hardware_fingerprint"):
                if not row.get(key) or row[key] != expected[key]:
                    raise ValueError(f"Baseline mismatch: {key}")
            new_identity = expected.get("execution_identities", {}).get((row["model"], row["seed"]))
            if "execution_identities" in expected and (
                not new_identity
                or row.get("execution_identity", {}).get("fingerprint") != new_identity
            ):
                raise ValueError(
                    "Missing or incompatible focused execution fingerprint; rerun Full baseline"
                )
            if (
                "execution_identities" not in expected
                and row.get("source_snapshot") != expected["source_snapshot"]
            ):
                raise ValueError("Benchmark code snapshot differs")
            config = row.get("config", {})
            recipe = expected["configs"][row["model"]]
            for key in (
                "model",
                "training",
                "model_options",
                "threads",
                "split_mode",
                "split_index",
                "split_seed",
            ):
                if config.get(key) != recipe.get(key):
                    raise ValueError(f"Baseline recipe mismatch: {row['model']}/{key}")
            if row.get("source_type") != expected["sources"][row["model"]]:
                raise ValueError("Model implementation provenance differs")
            software = row.get("software", {})
            if "execution_identities" not in expected and (
                software.get("torch") != expected["software"]["torch"]
                or software.get("pyg") != expected["software"]["pyg"]
            ):
                raise ValueError("Baseline software versions missing or different")
            if row.get("repository_commit") != expected.get("commits", {}).get(row["model"]):
                raise ValueError("Implementation commit differs")
            if not finite_nonnegative(row.get("training_time_sec")):
                raise ValueError("Missing measured training time")
        elapsed = metadata.get("elapsed_seconds")
        # Wall time must cover this exact matrix, not a sum of per-run or worker times.
        if (
            not finite_nonnegative(elapsed)
            or metadata.get("elapsed_scope") != "full_matrix"
            or not all(r.get("run_id") for r in rows)
            or sorted(metadata.get("run_ids", [])) != sorted(r["run_id"] for r in rows)
        ):
            elapsed = None
        return {
            "full_benchmark_baseline": "available",
            "training_seconds": sum(r["training_time_sec"] for r in rows),
            "elapsed_seconds": elapsed,
            "optimization_seconds": sum(
                r.get("optimization_seconds", r.get("optimization_time_sec", 0)) for r in rows
            )
            if all("optimization_seconds" in r or "optimization_time_sec" in r for r in rows)
            else None,
            "path": str(source.resolve()),
            "file_sha256": file_sha256(source),
            "reason": "Compatible measured runs; wall clock unavailable"
            if elapsed is None
            else "Compatible measured runs and full-matrix wall clock",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**unavailable, "reason": str(exc)}


def compute_accounting(training: float, elapsed: float, baseline: dict, records=None) -> dict:
    def reduction(reference, measured):
        return (
            100 * (reference - measured) / reference
            if reference is not None and reference > 0
            else None
        )

    if records is not None:
        optimization = sum(r.get("optimization_seconds", 0) for r in records)
        evaluation = sum(
            sum(
                r.get(k, 0)
                for k in (
                    "validation_seconds",
                    "pre_adaptation_evaluation_seconds",
                    "post_adaptation_evaluation_seconds",
                    "test_evaluation_seconds",
                )
            )
            for r in records
        )
        residual = max(0, elapsed - optimization - evaluation)
    train_reduction = reduction(baseline.get("training_seconds"), training)
    wall_reduction = reduction(baseline.get("elapsed_seconds"), elapsed)
    result = {
        "full_benchmark_baseline": baseline["full_benchmark_baseline"],
        "baseline_reason": baseline["reason"],
        "full_benchmark_training_seconds": baseline.get("training_seconds"),
        "progressive_training_seconds": training,
        "full_benchmark_elapsed_seconds": baseline.get("elapsed_seconds"),
        "progressive_elapsed_seconds": elapsed,
        "non_training_overhead_seconds": max(0.0, elapsed - training),
        "training_time_reduction_percent": train_reduction,
        "wall_clock_reduction_percent": wall_reduction,
        "status": (
            "progressive_overhead_exceeds_savings"
            if wall_reduction < 0
            else "no_wall_clock_change"
            if wall_reduction == 0
            else "wall_clock_savings"
        )
        if wall_reduction is not None
        else "wall_clock_comparison_unavailable",
    }

    if records is not None:
        result.update(
            optimization_seconds=optimization,
            evaluation_seconds=evaluation,
            other_execution_seconds=residual,
            total_benchmark_elapsed_seconds=elapsed,
            failed_attempt_elapsed_seconds=sum(
                r.get("failed_attempt_elapsed_seconds", 0) for r in records
            ),
            full_optimization_seconds=baseline.get("optimization_seconds"),
            optimization_reduction_percent=reduction(
                baseline.get("optimization_seconds"), optimization
            ),
            timing_note=(
                "Optimization + evaluation + other execution = elapsed. "
                "Failed-attempt elapsed and phase totals are diagnostic subsets, "
                "not additive categories."
            ),
        )
    return result
