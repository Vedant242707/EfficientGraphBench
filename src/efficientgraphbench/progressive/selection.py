"""Validation-only pruning with transparent dispersion and boundary protection."""

import math
from statistics import mean, stdev

from efficientgraphbench.profiling.events import observed


@observed("candidate_selection")
def select_candidates(
    records: list[dict],
    stage,
    minimum: int,
    promotion_margin: float = 0.0,
    uncertainty_aware: bool = True,
    expected_seeds=None,
) -> dict:
    grouped = {}
    for row in records:
        grouped.setdefault(row["model"], []).append(row)
    stats = {}
    for model, rows in grouped.items():
        if expected_seeds is not None:
            seeds = [r.get("seed") for r in rows]
            if set(seeds) != set(expected_seeds) or len(seeds) != len(set(expected_seeds)):
                continue
        scores = [r.get("validation_metric") for r in rows if r["status"] == "SUCCESS"]
        if len(scores) != len(rows) or not all(v is not None and math.isfinite(v) for v in scores):
            continue
        stats[model] = {
            "mean": mean(scores),
            "sd": stdev(scores) if len(scores) > 1 else 0.0,
            "successful_seed_count": len(scores),
        }
    ranking = sorted(stats, key=lambda model: (-stats[model]["mean"], model))
    retain = min(
        len(ranking),
        max(minimum, stage.retain_top_k or math.ceil(len(ranking) * stage.retain_fraction)),
    )
    if not ranking:
        return {}
    boundary = stats[ranking[retain - 1]]
    decisions = {}
    for index, model in enumerate(ranking):
        score = stats[model]
        reason = "below_stage_cutoff"
        if index < retain:
            reason = "within_stage_cutoff"
        elif promotion_margin > 0 and score["mean"] >= boundary["mean"] - promotion_margin:
            reason = "within_promotion_margin"
        elif (
            uncertainty_aware
            and score["successful_seed_count"] > 1
            and boundary["successful_seed_count"] > 1
            and score["mean"] + score["sd"] >= boundary["mean"] - boundary["sd"]
        ):
            reason = "uncertainty_overlap"
        decisions[model] = {
            "rank": index + 1,
            "score": score["mean"],
            "mean_validation_score": score["mean"],
            "validation_score_sd": score["sd"],
            "successful_seed_count": score["successful_seed_count"],
            "cutoff_mean": boundary["mean"],
            "cutoff_sd": boundary["sd"],
            "nominal_retain_count": retain,
            "promotion_margin": promotion_margin,
            "uncertainty_rule": "mean_plus_minus_sample_sd" if uncertainty_aware else "disabled",
            "candidate_status": "eliminated" if reason == "below_stage_cutoff" else "advanced",
            "reason": reason,
        }
    return decisions
