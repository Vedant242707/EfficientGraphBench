import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from efficientgraphbench import Benchmark, BenchmarkResult, BenchmarkResults, ScalingBenchmark
from efficientgraphbench.cli.app import app
from efficientgraphbench.datasets import InvalidDatasetError, load_dataset
from efficientgraphbench.environments.manager import environment_spec, inspect_environment
from efficientgraphbench.models.metadata import get_model


def test_public_dataset_and_cpu_run(tmp_path):
    data = load_dataset(
        "custom", nodes="examples/custom_graph/nodes.csv", edges="examples/custom_graph/edges.csv"
    )
    result = Benchmark.run_model(
        "gcn",
        dataset=data,
        epochs=1,
        device="cpu",
        latency_warmup=0,
        latency_repeats=1,
        output_dir=str(tmp_path),
    )
    assert isinstance(result, BenchmarkResult)
    assert result.status == "SUCCESS"
    assert 0 <= result.accuracy <= 1
    assert result.training_time >= 0
    assert result.record["efficientgraphbench_version"] == "0.1.0"
    assert result.record["source_snapshot"]


def test_api_failures_and_exports(tmp_path):
    results = Benchmark(
        dataset="custom",
        dataset_path="examples/custom_graph",
        models=["graphormer", "gcn"],
        seeds=[1, 2],
        epochs=1,
        device="cpu",
        latency_warmup=0,
        latency_repeats=1,
        output_dir=str(tmp_path),
    ).run()
    assert [r.status for r in results] == [
        "UNSUPPORTED_TASK",
        "SKIPPED_DETERMINISTIC_FAILURE",
        "SUCCESS",
        "SUCCESS",
    ]
    assert results[1].accuracy is None
    assert len(results.summary()) == 2
    for method, filename in [
        ("save_json", "a.json"),
        ("save_jsonl", "a.jsonl"),
        ("save_csv", "a.csv"),
        ("save_markdown", "a.md"),
    ]:
        path = getattr(results, method)(tmp_path / filename)
        assert path.stat().st_size > 0
    assert len(json.loads((tmp_path / "a.json").read_text())) == 4
    assert "UNSUPPORTED_TASK" in (tmp_path / "a.md").read_text(encoding="utf-8")


def test_scaling_api_latest_only(tmp_path):
    result = ScalingBenchmark(
        dataset=load_dataset("custom", dataset_path="examples/custom_graph"),
        model="gcn",
        sizes=[6, 12],
        seeds=[42],
        epochs=1,
        device="cpu",
        latency_warmup=0,
        latency_repeats=1,
        output_dir=str(tmp_path),
    ).run()
    assert len(result) == 1
    assert result[0].num_nodes == 12
    assert len(json.loads(result.all_results_path.read_text())) == 2


def test_environment_cache_and_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("EGBENCH_HOME", str(tmp_path))
    spec = environment_spec("graphgps")
    assert str(tmp_path) in spec["python"]
    assert not spec["validated"]
    assert inspect_environment("graphgps")["status"] == "ENVIRONMENT_NOT_INSTALLED"
    assert get_model("gatv2")["supported_tasks"] == ["node_classification"]
    assert get_model("graphormer")["status"] == "UNSUPPORTED_TASK"
    assert CliRunner().invoke(app, ["models", "graphormer"]).exit_code == 0
    for command in ("setup", "environments", "report", "scaling"):
        assert CliRunner().invoke(app, [command, "--help"]).exit_code == 0


def test_custom_error_names_detected_columns(tmp_path):
    nodes = Path("examples/custom_graph/nodes.csv")
    edges = tmp_path / "relationships.csv"
    edges.write_text("source_node,target_node\na,b\n")
    with pytest.raises(InvalidDatasetError, match="source_node"):
        load_dataset("custom", nodes=nodes, edges=edges)


def test_results_never_mix_comparison_groups():
    result = BenchmarkResults(
        [
            {
                "model": "gcn",
                "dataset": "custom",
                "comparison_group": group,
                "status": "SUCCESS",
                "accuracy": accuracy,
            }
            for group, accuracy in [("a", 0.1), ("b", 0.9)]
        ]
    )
    assert len(result.summary()) == 2


def test_setup_verifies_existing_reference_without_reinstall(tmp_path, monkeypatch):
    from efficientgraphbench.environments import manager

    monkeypatch.setenv("EGBENCH_HOME", str(tmp_path))
    spec = manager.environment_spec("sgformer")
    Path(spec["python"]).parent.mkdir(parents=True)
    Path(spec["python"]).touch()
    Path(spec["repository"]).mkdir(parents=True)
    calls = []

    def verify(name, verify=False):
        calls.append((name, verify))
        return {**spec, "validated": True, "status": "READY"}

    monkeypatch.setattr(manager, "inspect_environment", verify)
    monkeypatch.setattr(
        manager.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected reinstall")
    )
    assert manager.setup_environment("sgformer")["status"] == "READY"
    assert calls == [("sgformer", True)]
    assert (tmp_path / "environments/sgformer/metadata.json").exists()


def test_missing_reference_is_failure_result(tmp_path, monkeypatch):
    monkeypatch.setenv("EGBENCH_HOME", str(tmp_path / "cache"))
    result = Benchmark.run_model(
        "graphgps",
        dataset="custom",
        dataset_path="examples/custom_graph",
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    assert result.status == "ENVIRONMENT_NOT_INSTALLED"
    assert result.accuracy is None
    assert "setup graphgps" in result.failure_reason


def test_windows_hardware_identity_ignores_python_marketing_label(monkeypatch):
    from types import SimpleNamespace

    from efficientgraphbench.benchmark import identity

    monkeypatch.setattr(identity.sys, "platform", "win32")
    monkeypatch.setattr(
        identity.sys,
        "getwindowsversion",
        lambda: SimpleNamespace(major=10, minor=0, build=26200),
        raising=False,
    )
    monkeypatch.setattr(identity.platform, "platform", lambda: "Windows-10-10.0.26200-SP0")
    first_info, first = identity.physical_hardware("cpu")
    monkeypatch.setattr(identity.platform, "platform", lambda: "Windows-11-10.0.26200-SP0")
    second_info, second = identity.physical_hardware("cpu")
    assert first == second
    assert first_info["os"] != second_info["os"]
    monkeypatch.setattr(identity.platform, "node", lambda: "different-machine")
    assert identity.physical_hardware("cpu")[1] != first
    monkeypatch.setattr(
        identity.sys, "getwindowsversion", lambda: SimpleNamespace(major=10, minor=0, build=99999)
    )
    assert identity.physical_hardware("cpu")[1] != second
