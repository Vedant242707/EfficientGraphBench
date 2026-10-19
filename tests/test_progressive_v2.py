import copy
import json

import pytest

from efficientgraphbench.progressive.adaptation import recovery_status, transition_metrics
from efficientgraphbench.progressive.baseline import compute_accounting, load_full_baseline
from efficientgraphbench.progressive.config import FidelityStage, ProgressiveConfig
from efficientgraphbench.progressive.selection import select_candidates


def test_signed_transition_semantics_and_pre_post():
    m = transition_metrics(0.82, 0.8, 0.61, 0.78, 0.75)
    assert m["post_expansion_delta"] == pytest.approx(-0.02)
    assert m["old_node_performance_delta"] == pytest.approx(-0.04)
    assert m["old_node_forgetting"] == pytest.approx(0.04)
    assert m["old_node_adaptation_delta"] == pytest.approx(-0.02)
    assert m["new_node_adaptation_gain"] == pytest.approx(0.14)
    assert m["old_new_gap"] == pytest.approx(0.03)
    improved = transition_metrics(0.82, 0.84, 0.61, 0.85, 0.75)
    assert improved["old_node_performance_delta"] == pytest.approx(0.03)
    assert improved["old_node_forgetting"] == 0
    assert improved["post_expansion_delta"] > 0
    initial = transition_metrics(None, None, None, None, 0.5)
    assert initial["old_node_forgetting"] is None
    assert initial["new_node_adaptation_gain"] is None


@pytest.mark.parametrize(
    "old,pre,target,recovered,expected",
    [
        (0.8, 0.77, 0.76, True, "recovery_not_required"),
        (0.8, 0.70, 0.76, True, "recovered"),
        (0.8, 0.70, 0.76, False, "not_recovered"),
        (None, None, None, False, "not_applicable"),
    ],
)
def test_all_recovery_states(old, pre, target, recovered, expected):
    assert recovery_status(old, pre, target, recovered) == expected


def rows(scores):
    return [
        dict(
            model=model, seed=i, status="SUCCESS", validation_metric=value, test_accuracy=1 - value
        )
        for model, values in scores.items()
        for i, value in enumerate(values)
    ]


def test_margin_uncertainty_and_validation_only():
    stage = FidelityStage("first", 0.5, 2, retain_top_k=1)
    sample = rows({"a": [0.8, 0.8], "b": [0.79, 0.79], "c": [0.5, 0.5]})
    strict = select_candidates(sample, stage, 1, uncertainty_aware=False)
    assert strict["b"]["candidate_status"] == "eliminated"
    protected = select_candidates(sample, stage, 1, promotion_margin=0.011)
    assert protected["b"]["reason"] == "within_promotion_margin"
    assert protected["c"]["candidate_status"] == "eliminated"
    overlap = rows({"a": [0.7, 0.9], "b": [0.68, 0.88], "c": [0.4, 0.5]})
    decisions = select_candidates(overlap, stage, 1)
    assert decisions["b"]["reason"] == "uncertainty_overlap"
    assert decisions["b"]["successful_seed_count"] == 2
    assert decisions["b"]["validation_score_sd"] > 0
    assert decisions["c"]["candidate_status"] == "eliminated"
    for row in overlap:
        row["test_accuracy"] = 999
    assert select_candidates(overlap, stage, 1) == decisions
    assert select_candidates(list(reversed(overlap)), stage, 1) == decisions


@pytest.mark.parametrize(
    "baseline_time,elapsed,reduction,status",
    [
        (100, 50, 50, "wall_clock_savings"),
        (100, 100, 0, "no_wall_clock_change"),
        (100, 120, -20, "progressive_overhead_exceeds_savings"),
        (None, 120, None, "wall_clock_comparison_unavailable"),
    ],
)
def test_compute_positive_zero_negative_unavailable(baseline_time, elapsed, reduction, status):
    b = {
        "full_benchmark_baseline": "available" if baseline_time else "unavailable",
        "reason": "test",
        "training_seconds": baseline_time,
        "elapsed_seconds": baseline_time,
    }
    result = compute_accounting(40, elapsed, b)
    assert result["wall_clock_reduction_percent"] == reduction
    assert result["status"] == status
    assert result["non_training_overhead_seconds"] == elapsed - 40
    assert result["training_time_reduction_percent"] == (60 if baseline_time else None)


def baseline_fixture():
    recipe = {
        "model": {"name": "gcn"},
        "training": {"epochs": 100},
        "model_options": {},
        "threads": 4,
        "split_mode": "official",
        "split_index": 0,
        "split_seed": 0,
    }
    expected = {
        "models": ["gcn"],
        "seeds": [1, 2],
        "canonical_graph_hash": "graph",
        "split_hash": "split",
        "hardware_fingerprint": "hw",
        "source_snapshot": {"code": "hash"},
        "software": {"torch": "t", "pyg": "p"},
        "configs": {"gcn": recipe},
        "sources": {"gcn": "library"},
    }
    records = [
        {
            "model": "gcn",
            "seed": seed,
            "run_id": str(seed),
            "status": "SUCCESS",
            "config": recipe,
            "source_type": "library",
            "training_time_sec": 10.0,
            "canonical_graph_hash": "graph",
            "split_hash": "split",
            "hardware_fingerprint": "hw",
            "source_snapshot": {"code": "hash"},
            "software": {"torch": "t", "pyg": "p"},
        }
        for seed in (1, 2)
    ]
    return expected, records


def test_saved_baseline_verification(tmp_path):
    expected, records = baseline_fixture()
    path = tmp_path / "full.json"
    path.write_text(
        json.dumps(
            {
                "records": records,
                "metadata": {
                    "elapsed_seconds": 30,
                    "elapsed_scope": "full_matrix",
                    "run_ids": ["1", "2"],
                },
            }
        )
    )
    result = load_full_baseline(str(path), expected)
    assert result["full_benchmark_baseline"] == "available"
    assert result["training_seconds"] == 20
    assert result["elapsed_seconds"] == 30
    path.write_text(json.dumps(records))
    assert load_full_baseline(str(path), expected)["elapsed_seconds"] is None
    for key, value in [
        ("hardware_fingerprint", "different"),
        ("split_hash", "wrong"),
        ("canonical_graph_hash", "wrong"),
        ("benchmark_mode", "progressive_multi_fidelity"),
        ("source_snapshot", {}),
        ("status", "OOM"),
    ]:
        changed = copy.deepcopy(records)
        changed[0][key] = value
        path.write_text(json.dumps(changed))
        assert load_full_baseline(str(path), expected)["full_benchmark_baseline"] == "unavailable"
    assert load_full_baseline(None, expected)["full_benchmark_baseline"] == "unavailable"


def test_config_rejects_unsupported_audit_and_estimates():
    with pytest.raises(TypeError):
        ProgressiveConfig(audit=True)
    with pytest.raises(TypeError):
        ProgressiveConfig(full_benchmark_seconds=100)
    with pytest.raises(ValueError, match="promotion_margin"):
        ProgressiveConfig(stages=[FidelityStage("full", 1, 1)], promotion_margin=-0.01).validate()


def test_report_separates_partial_scores_and_finalists(tmp_path):
    from efficientgraphbench.progressive.results import ProgressiveResults

    def record(model, fraction, state, score):
        return {
            "model": model,
            "seed": 1,
            "status": "SUCCESS",
            "candidate_status": state,
            "validation_metric": score,
            "training_time_sec": 1.0,
            "fidelity": {
                "stage_index": 0 if fraction < 1 else 1,
                "stage_name": "stage",
                "graph_fraction": fraction,
                "additional_epochs": 2,
            },
            "adaptation": {
                "overall_metric": score,
                "recovery_status": "not_applicable",
                "previous_stage_score": None,
            },
            "promotion": {
                "reason": "below_stage_cutoff" if fraction < 1 else "completed_full_graph"
            },
            "test_metrics": {"accuracy": score, "f1_macro": score, "f1_micro": score}
            if fraction == 1
            else None,
        }

    r = ProgressiveResults(
        [record("gcn", 1, "advanced", 0.7), record("mlp", 0.5, "eliminated", 0.99)],
        tmp_path,
        {"seeds": [1], "config": {"stages": []}, "compute": {}},
    )
    report = r.show_all(display=False).read_text(encoding="utf8")
    finalists = report.split("<h2>Full-graph finalists</h2>")[1].split(
        "<h2>Early-eliminated candidates</h2>"
    )[0]
    partial = report.split("<h2>Early-eliminated candidates</h2>")[1].split("<h3>Failed, skipped")[
        0
    ]
    assert "GCN" in finalists and "MLP" not in finalists
    assert "MLP" in partial and "GCN" not in partial
    assert "0.990" not in finalists
    assert "Methodology and metric glossary" in report
    # One successful seed is not enough when two were requested.
    r.metadata["seeds"] = [1, 2]
    assert not r.summary()["Finalist"].any()
