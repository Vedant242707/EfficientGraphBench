"""Recommendations are filters over measured, comparable primary results."""

from collections import defaultdict
from statistics import mean

from efficientgraphbench.results.status import is_success


def recommend(records, minimum_accuracy, maximum_latency_ms, maximum_gpu_mib, group=None):
    eligible = [
        r
        for r in records
        if is_success(r)
        and r.get("schema_version", 0) >= 4
        and r.get("category") == "primary"
        and r.get("source_type") in {"official_repo", "library"}
    ]
    groups = {r["comparison_group"] for r in eligible}
    if group is None and len(groups) > 1:
        raise ValueError("Multiple hardware/dataset/split groups; specify --group")
    if group is not None:
        eligible = [r for r in eligible if r["comparison_group"] == group]
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["model"]].append(row)
    recommendations = []
    for model, rows in grouped.items():
        # Require every seed in this comparison to have succeeded for this model.
        expected = {
            r["seed"] for r in records if r.get("comparison_group") == rows[0]["comparison_group"]
        }
        if {r["seed"] for r in rows} != expected:
            continue
        if any(r.get("peak_gpu_reserved_mib") is None for r in rows):
            continue
        accuracy = mean(r["accuracy"] for r in rows)
        latency = mean(r["inference_latency_median_ms"] for r in rows)
        memory = max(r["peak_gpu_reserved_mib"] for r in rows)
        if (
            accuracy >= minimum_accuracy
            and latency <= maximum_latency_ms
            and memory <= maximum_gpu_mib
        ):
            recommendations.append(
                {
                    "model": model,
                    "accuracy": accuracy,
                    "latency_ms": latency,
                    "gpu_reserved_mib": memory,
                    "seeds": len(expected),
                    "group": rows[0]["comparison_group"],
                }
            )
    return sorted(recommendations, key=lambda r: (-r["accuracy"], r["latency_ms"]))
