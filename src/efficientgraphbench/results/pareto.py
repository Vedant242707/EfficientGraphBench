"""Exact Pareto analysis of measured configurations, without retraining."""

import math
from collections import defaultdict
from statistics import mean, stdev

from efficientgraphbench.benchmark.interchange import fingerprint
from efficientgraphbench.results.status import status_of


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def pareto_frontier(
    records,
    *,
    quality="f1_macro",
    memory="auto",
    include_training=False,
    minimum_quality=0.0,
    minimum_accuracy=0.0,
    maximum_latency_ms=None,
    maximum_memory_mib=None,
    minimum_runs=1,
    weights=None,
):
    """Return separate frontiers for comparable graphs/hardware/budgets.

    All matched seed/split attempts must succeed with finite objective metrics.
    Ties are retained. No significance claim is made from exact mean dominance.
    CPU auto mode uses quality/latency only; CPU RSS can be explicitly selected.
    Optional balanced selection uses normalized losses and explicit weights.
    """
    if hasattr(records, "records"):
        records = records.records
    records = list(records)
    if quality not in {"accuracy", "f1_macro"} or memory not in {"auto", "gpu", "cpu", "none"}:
        raise ValueError("quality: accuracy/f1_macro; memory: auto/gpu/cpu/none")
    if type(minimum_runs) is not int or minimum_runs < 1:
        raise ValueError("minimum_runs must be a positive integer")
    for value in (minimum_quality, minimum_accuracy):
        if not _finite(value) or not 0 <= value <= 1:
            raise ValueError("Minimum scores must be finite fractions in [0,1]")
    for value in (maximum_latency_ms, maximum_memory_mib):
        if value is not None and (not _finite(value) or value <= 0):
            raise ValueError("Resource limits must be finite and positive")
    cohorts, excluded = defaultdict(list), []
    cohort_settings = {}
    for row in records:
        reason = None
        if row.get("source_type") not in {"library", "official_repo"}:
            reason = "Missing or unsupported implementation provenance"
        elif any(
            not row.get(k) for k in ("graph_content_hash", "hardware_fingerprint", "split_hash")
        ):
            reason = "Missing graph/hardware/split identity; comparison cannot be verified"
        elif row.get("seed") is None:
            reason = "Missing seed identity"
        elif not row.get("profiler_settings") or not (row.get("config") or {}).get("training"):
            reason = "Missing measurement settings or training budget"
        if reason:
            excluded.append(
                {
                    "model": row.get("model"),
                    "run_id": row.get("run_id"),
                    "status": status_of(row),
                    "reason": reason,
                }
            )
            continue
        cfg = row.get("config") or {}
        training = cfg.get("training") or {}
        identity = {
            k: row.get(k)
            for k in (
                "dataset",
                "graph_content_hash",
                "hardware_fingerprint",
                "device",
                "num_nodes",
                "num_edges",
                "profiler_settings",
                "source_snapshot",
                "feature_preprocessing",
                "edge_policy",
                "batch_size",
            )
        }
        identity.update(
            maximum_epochs=row.get("maximum_epochs", training.get("epochs")),
            early_stopping_patience=row.get(
                "early_stopping_patience", training.get("early_stopping_patience")
            ),
            split_mode=cfg.get("split_mode"),
        )
        group_id = fingerprint(identity)[:16]
        cohorts[group_id].append(row)
        cohort_settings[group_id] = identity
    result = {
        "groups": [],
        "excluded": excluded,
        "note": "Measured configuration trade-offs, not universal model rankings. "
        "Exact mean dominance is descriptive, not statistical significance. "
        "Seeds/folds must match; software and model recipes may differ.",
    }
    for group_id, rows in sorted(cohorts.items()):
        first = rows[0]
        memory_key = {
            "gpu": "peak_gpu_reserved_mib",
            "cpu": "peak_cpu_memory_mb",
            "none": None,
        }.get(memory)
        if memory == "auto":
            memory_key = (
                "peak_gpu_reserved_mib" if str(first.get("device", "")).startswith("cuda") else None
            )
        if maximum_memory_mib is not None and memory_key is None:
            raise ValueError("A memory limit requires a measured memory objective (gpu or cpu)")
        objectives = {"quality": "maximize", "latency_ms": "minimize"}
        if memory_key:
            objectives["memory_mib"] = "minimize"
        if include_training:
            objectives["training_time_sec"] = "minimize"
        if weights is not None and (
            set(weights) != set(objectives)
            or any(not _finite(v) or v < 0 for v in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                f"weights must cover {list(objectives)} with nonnegative values and positive sum"
            )
        candidates = defaultdict(list)
        expected = {(r["seed"], r["split_hash"]) for r in rows}
        for row in rows:
            cfg = row.get("config") or {}
            recipe = fingerprint(
                {
                    "model": row.get("model"),
                    "settings": cfg.get("model"),
                    "training": cfg.get("training"),
                    "options": cfg.get("model_options"),
                    "software": row.get("software"),
                    "source": row.get("repository_commit"),
                    "environment": row.get("environment"),
                }
            )[:12]
            candidates[(row.get("model"), recipe)].append(row)
        group = {
            "group": group_id,
            "comparison_settings": cohort_settings[group_id],
            "dataset": first.get("dataset"),
            "num_nodes": first.get("num_nodes"),
            "device": first.get("device"),
            "hardware_fingerprint": first.get("hardware_fingerprint"),
            "quality_metric": quality,
            "memory_metric": memory_key,
            "objectives": objectives,
            "candidates": [],
            "excluded": [],
            "recommendations": {},
        }
        for (model, recipe), runs in sorted(candidates.items()):
            pairs = [(r["seed"], r["split_hash"]) for r in runs]
            reason = None
            required = {quality, "accuracy", "inference_latency_median_ms", "training_time_sec"}
            if memory_key:
                required.add(memory_key)
            if any(status_of(r) != "SUCCESS" for r in runs):
                reason = "Contains failed or skipped runs"
            elif any(
                status_of(r) != "SUCCESS"
                and r.get("model") == model
                and r.get("dataset") == first.get("dataset")
                and not r.get("graph_content_hash")
                and r.get("num_nodes") in (None, first.get("num_nodes"))
                and (not r.get("batch_id") or r.get("batch_id") == runs[0].get("batch_id"))
                for r in records
            ):
                reason = (
                    "A failed/skipped attempt lacks graph identity; completeness cannot be verified"
                )
            elif len(set(pairs)) != len(pairs):
                reason = "Duplicate seed/split attempts; select one experiment batch"
            elif set(pairs) != expected:
                reason = "Incomplete or mismatched seed/split coverage"
            elif len(runs) < minimum_runs:
                reason = f"Requires at least {minimum_runs} matched runs"
            elif any(not _finite(r.get(k)) or r[k] < 0 for r in runs for k in required):
                reason = "Missing or invalid objective metrics; rerun with current metrics"
            elif any(r[quality] > 1 or r["accuracy"] > 1 for r in runs):
                reason = "Classification metrics must be fractions in [0,1]"
            if reason:
                group["excluded"].append({"model": model, "candidate": recipe, "reason": reason})
                continue
            scores = [r[quality] for r in runs]
            candidate = {
                "model": model,
                "candidate": recipe,
                "runs": len(runs),
                "quality": mean(scores),
                "quality_sd": stdev(scores) if len(scores) > 1 else 0,
                "accuracy": mean(r["accuracy"] for r in runs),
                "latency_ms": mean(r["inference_latency_median_ms"] for r in runs),
                "training_time_sec": mean(r["training_time_sec"] for r in runs),
                "memory_mib": max(r[memory_key] for r in runs) if memory_key else None,
                "run_ids": [r.get("run_id") for r in runs],
            }
            breaches = []
            if candidate["quality"] < minimum_quality:
                breaches.append("quality below minimum")
            if candidate["accuracy"] < minimum_accuracy:
                breaches.append("accuracy below minimum")
            if maximum_latency_ms is not None and candidate["latency_ms"] > maximum_latency_ms:
                breaches.append("latency exceeds budget")
            if maximum_memory_mib is not None and candidate["memory_mib"] > maximum_memory_mib:
                breaches.append("memory exceeds budget")
            candidate.update(
                qualified=not breaches,
                constraint_failures=breaches,
                frontier=False,
                dominated_by=[],
            )
            group["candidates"].append(candidate)
        qualified = [c for c in group["candidates"] if c["qualified"]]

        def dominates(a, b):
            better = [
                a[k] >= b[k] if direction == "maximize" else a[k] <= b[k]
                for k, direction in objectives.items()
            ]
            strict = any(a[k] != b[k] for k in objectives)
            return all(better) and strict

        for candidate in qualified:
            candidate["dominated_by"] = [
                c["candidate"] for c in qualified if dominates(c, candidate)
            ]
            candidate["frontier"] = not candidate["dominated_by"]
        if qualified:
            for label, key, maximize in (
                ("best_quality", "quality", True),
                ("fastest_inference", "latency_ms", False),
                ("fastest_training", "training_time_sec", False),
            ):
                best = (max if maximize else min)(c[key] for c in qualified)
                group["recommendations"][label] = [
                    c["candidate"] for c in qualified if c[key] == best
                ]
            if memory_key:
                best = min(c["memory_mib"] for c in qualified)
                group["recommendations"]["lowest_memory"] = [
                    c["candidate"] for c in qualified if c["memory_mib"] == best
                ]
            group["recommendations"]["overall_shortlist"] = [
                c["candidate"] for c in qualified if c["frontier"]
            ]
            if weights is not None:
                for c in qualified:
                    loss = 0.0
                    for key, direction in objectives.items():
                        lo, hi = min(x[key] for x in qualified), max(x[key] for x in qualified)
                        scaled = (c[key] - lo) / (hi - lo) if hi > lo else 0
                        if direction == "maximize" and hi > lo:
                            scaled = 1 - scaled
                        loss += weights[key] * scaled / sum(weights.values())
                    c["balanced_loss"] = loss
                front = [c for c in qualified if c["frontier"]]
                best = min(c["balanced_loss"] for c in front)
                group["recommendations"]["balanced"] = [
                    c["candidate"] for c in front if c["balanced_loss"] == best
                ]
                group["weights"] = dict(weights)
        result["groups"].append(group)
    return result
