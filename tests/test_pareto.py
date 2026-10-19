import copy
import json

import pytest
from typer.testing import CliRunner

from efficientgraphbench import BenchmarkResults
from efficientgraphbench.cli.main import app
from efficientgraphbench.results.dashboard import load_analysis_results
from efficientgraphbench.results.pareto import pareto_frontier


def row(model, quality, latency, memory=100, **kw):
    return dict(
        {
            "model": model,
            "dataset": "texas",
            "status": "SUCCESS",
            "seed": 1,
            "run_id": model,
            "source_type": "library",
            "graph_content_hash": "graph",
            "hardware_fingerprint": "gpu",
            "split_hash": "split1",
            "device": "cuda",
            "num_nodes": 100,
            "num_edges": 200,
            "accuracy": quality,
            "f1_macro": quality,
            "precision": quality,
            "recall": quality,
            "f1": quality,
            "f1_micro": quality,
            "inference_latency_median_ms": latency,
            "training_time_sec": 5,
            "peak_gpu_reserved_mib": memory,
            "peak_gpu_allocated_mib": memory - 10,
            "peak_cpu_memory_mb": 200,
            "comparison_group": "comparison",
            "config": {"training": {"epochs": 20}},
            "profiler_settings": {"repeats": 10},
        },
        **kw,
    )


def test_frontier_tradeoffs_ties_constraints_and_recommendations():
    records = [row("gcn", 0.9, 10), row("gat", 0.8, 5), row("mlp", 0.7, 15), row("sgc", 0.9, 10)]
    group = pareto_frontier(records)["groups"][0]
    assert {c["model"] for c in group["candidates"] if c["frontier"]} == {"gcn", "gat", "sgc"}
    assert next(c for c in group["candidates"] if c["model"] == "mlp")["dominated_by"]
    assert len(group["recommendations"]["best_quality"]) == 2
    filtered = pareto_frontier(records, minimum_quality=0.85)["groups"][0]
    assert {c["model"] for c in filtered["candidates"] if c["qualified"]} == {"gcn", "sgc"}
    balanced = pareto_frontier(
        records, weights={"quality": 0.8, "latency_ms": 0.1, "memory_mib": 0.1}
    )
    assert balanced["groups"][0]["recommendations"]["balanced"]


def test_matched_folds_and_incomplete_or_failed_runs():
    a, b = row("gcn", 0.8, 10), row("gat", 0.7, 5)
    a2 = dict(a, seed=2, split_hash="split2", run_id="gcn2")
    b2 = dict(b, seed=2, split_hash="split2", run_id="gat2")
    group = pareto_frontier([a, b, a2, b2])["groups"][0]
    assert len(group["candidates"]) == 2
    assert all(c["runs"] == 2 for c in group["candidates"])
    for bad in (None, dict(b2, status="OOM"), dict(b2, split_hash="wrong")):
        group = pareto_frontier([a, b, a2] + ([bad] if bad else []))["groups"][0]
        assert "gat" in {x["model"] for x in group["excluded"]}
    missing = dict(b2, status="SKIPPED_AFTER_OOM", graph_content_hash=None)
    group = pareto_frontier([a, b, missing])["groups"][0]
    assert "gat" in {x["model"] for x in group["excluded"]}


def test_separate_hardware_graph_budget_and_explicit_cpu_behavior():
    a = row("gcn", 0.8, 10)
    variants = [
        dict(a, hardware_fingerprint="other"),
        dict(a, graph_content_hash="other"),
        dict(a, config={"training": {"epochs": 200}}),
    ]
    assert len(pareto_frontier([a, *variants])["groups"]) == 4
    cpu = dict(a, device="cpu", peak_gpu_reserved_mib=None)
    group = pareto_frontier([cpu])["groups"][0]
    assert set(group["objectives"]) == {"quality", "latency_ms"}
    assert pareto_frontier([cpu], memory="gpu")["groups"][0]["excluded"]
    assert (
        pareto_frontier([cpu], memory="cpu")["groups"][0]["memory_metric"] == "peak_cpu_memory_mb"
    )
    with pytest.raises(ValueError):
        pareto_frontier([cpu], maximum_memory_mib=500)


def test_invalid_legacy_duplicate_and_options():
    a = row("gcn", 0.8, 10)
    assert pareto_frontier([dict(a, f1_macro=None)])["groups"][0]["excluded"]
    assert pareto_frontier([dict(a, hardware_fingerprint=None)])["excluded"]
    assert pareto_frontier([a, a])["groups"][0]["excluded"]
    for options in (
        {"minimum_quality": float("nan")},
        {"quality": "typo"},
        {"minimum_runs": 0},
        {"weights": {"quality": -1}},
        {"maximum_latency_ms": -1},
    ):
        with pytest.raises(ValueError):
            pareto_frontier([a], **options)
    assert pareto_frontier([])["groups"] == []


def test_combined_report_and_cli_no_retraining(tmp_path):
    pytest.importorskip("matplotlib")
    records = [row("gcn", 0.9, 10), row("mlp", 0.8, 5)]
    original = copy.deepcopy(records)
    results = BenchmarkResults(records)
    report = results.show_all(tmp_path / "api", display=False)
    content = report["report"].read_text(encoding="utf-8")
    assert "Classification" in content and "Macro-F1" in content and "Pareto Comparison" in content
    assert "data:image/png;base64," in content
    assert "Quality: f1_macro" not in content
    assert "Peak GPU reserved memory — lower is better" in content
    assert "giving each class equal weight" in content
    assert "Pareto trade-offs" not in content
    assert {p.stem for p in report["plots"]} >= {
        "accuracy_vs_latency",
        "gpu_memory_allocation",
        "nodes_vs_training_time",
    }
    assert "Selection criteria" in content
    assert any(p.stem == "pareto_frontier" for p in report["plots"])
    assert records == original
    source = tmp_path / "runs.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    command = CliRunner().invoke(
        app, ["analyze", "--input", str(source), "--output-dir", str(tmp_path / "cli")]
    )
    assert command.exit_code == 0, command.output
    assert (tmp_path / "cli" / "analysis.html").exists()
    command = CliRunner().invoke(
        app, ["pareto", "--input", str(source), "--output", str(tmp_path / "pareto.json")]
    )
    assert command.exit_code == 0, command.output


def test_scaling_loader_retains_all_seeds_at_final_size(tmp_path):
    rows = [
        row("gcn", 0.8, 10, seed=s, run_id=f"{n}-{s}", num_nodes=n)
        for n in (10, 20)
        for s in (1, 2)
    ]
    path = tmp_path / "scaling-results.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    result = load_analysis_results(path)
    assert len(result.records) == 2
    assert {r["num_nodes"] for r in result.records} == {20}
    assert result.all_results_path == path


def test_failures_once_while_successful_statistics_remain_visible(tmp_path):
    records = [
        row("gcn", 0.8, 10),
        row("graphormer", 0.8, 10, status="OOM", failure_reason="Insufficient GPU memory"),
    ]
    records += [
        row("graphormer", 0.8, 10, seed=i, status="SKIPPED_AFTER_OOM") for i in range(2, 11)
    ]
    records += [
        row("gat", 0.9, 5),
        row("gat", 0.9, 5, seed=2, status="RUNTIME_ERROR", failure_reason="Worker stopped"),
    ]
    original = copy.deepcopy(records)
    results = BenchmarkResults(records)
    report = results.show_all(tmp_path, display=False)
    content = report["report"].read_text(encoding="utf-8")
    assert content.count("Graphormer") == 1
    assert content.count("GAT") >= 3
    assert "Insufficient GPU memory" in content
    assert "Statistics use successful runs only" in content
    assert "color:#163b50!important" in content
    assert "SKIPPED_AFTER_OOM" not in content
    markdown = results.save_markdown(tmp_path / "results.md").read_text(encoding="utf-8")
    assert markdown.count("Graphormer") == 1
    assert records == original


def test_group_notes_explain_actual_early_stopping_difference():
    from efficientgraphbench.results.dashboard import group_setup_notes

    records = [
        row("sgformer", 0.6, 20, maximum_epochs=200, early_stopping_patience=200),
        row("gcn", 0.5, 10, maximum_epochs=200, early_stopping_patience=None),
    ]
    groups = pareto_frontier(records)["groups"]
    sg = next(g for g in groups if g["candidates"][0]["model"] == "sgformer")
    notes = " ".join(group_setup_notes(sg, groups))
    assert "different early-stopping rule" in notes
    assert "cannot meaningfully shorten training" in notes
    assert "best-model labels are not shown" in notes
    records[0]["maximum_epochs"] = 500
    groups = pareto_frontier(records)["groups"]
    sg = next(g for g in groups if g["candidates"][0]["model"] == "sgformer")
    assert "cannot meaningfully shorten" not in " ".join(group_setup_notes(sg, groups))
