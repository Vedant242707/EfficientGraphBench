import json
import sys

import pytest
import torch
from torch_geometric.data import Data

from efficientgraphbench.benchmark.execution import ExternalEnvironmentRunner, RunnerError
from efficientgraphbench.benchmark.interchange import export_graph
from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.config import BenchmarkConfig, ModelConfig, TrainingConfig
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.models.registry import build_model


def bundle():
    return DatasetBundle(
        Data(
            x=torch.eye(6),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            y=torch.arange(6) % 2,
            train_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
            val_mask=torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
            test_mask=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        ),
        2,
    )


def test_canonical_splits_stable_and_sensitive(tmp_path):
    graph = bundle()
    first = export_graph(graph, tmp_path / "a")
    second = export_graph(graph, tmp_path / "b")
    assert first["split_hash"] == second["split_hash"]
    assert first["canonical_graph_hash"] == second["canonical_graph_hash"]
    graph.data.train_mask[0] = False
    third = export_graph(graph, tmp_path / "c")
    assert third["split_hash"] != first["split_hash"]


@pytest.mark.parametrize("model", ["graphgps", "graphormer", "sgformer"])
def test_primary_never_imports_adapted(model):
    with pytest.raises(ValueError, match="orchestrator"):
        build_model(ModelConfig(name=model), 6, 2)


def test_graphormer_reference_task_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset", lambda *a, **k: bundle()
    )
    result = run_benchmark(
        BenchmarkConfig(
            model=ModelConfig(name="graphormer"), device="cpu", output_dir=str(tmp_path)
        )
    )
    assert result["status"] == "UNSUPPORTED_TASK"
    assert result["num_nodes"] == 6
    assert result["implementation_limit"] is None
    assert result["split_hash"]


@pytest.mark.parametrize("mismatch", ["split_hash", "hardware_fingerprint", "model_id", "seed"])
def test_external_rejects_identity_mismatch(tmp_path, mismatch):
    worker = tmp_path / "fake.py"
    worker.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "r=json.loads(Path(sys.argv[1]).read_text())\n"
        f"r[{mismatch!r}]='wrong'\n"
        "r.update(status='SUCCESS',accuracy=0.5,validation_accuracy=0.5)\n"
        "Path(sys.argv[2]).write_text(json.dumps(r))\n",
        encoding="utf-8",
    )
    request = dict(
        run_id="r",
        model_id="graphgps",
        seed=42,
        split_hash="split",
        canonical_graph_hash="graph",
        hardware_fingerprint="hardware",
    )
    with pytest.raises(RunnerError, match="mismatch|hardware"):
        ExternalEnvironmentRunner([sys.executable, str(worker)], tmp_path / "run").run(request)


def test_external_missing_dependency_and_timeout(tmp_path):
    with pytest.raises(RunnerError) as error:
        ExternalEnvironmentRunner([str(tmp_path / "missing.exe")], tmp_path / "missing").run({})
    assert error.value.status == "DEPENDENCY_ERROR"
    worker = tmp_path / "sleep.py"
    worker.write_text("import time; time.sleep(10)")
    with pytest.raises(RunnerError, match="timeout"):
        ExternalEnvironmentRunner([sys.executable, str(worker)], tmp_path / "timeout", 0.1).run({})


def test_model_budget_differences_share_external_group(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset", lambda *a, **k: bundle()
    )
    config = BenchmarkConfig(
        device="cpu",
        output_dir=str(tmp_path),
        training=TrainingConfig(epochs=1, latency_warmup=0, latency_repeats=1),
    )
    first = run_benchmark(config)
    config.model = ModelConfig(name="graphsage", hidden_dim=16)
    config.training.lr = 0.001
    second = run_benchmark(config)
    assert first["comparison_group"] == second["comparison_group"]
    assert first["parameter_count"] != second["parameter_count"]
    assert first["peak_gpu_reserved_mib"] is None
    json.dumps(second, allow_nan=False)


def test_reference_options_reject_invalid_or_unknown_values():
    for options in ({"lr": -1}, {"hidden_dim": 62}, {"dropout": 1}, {"invented": True}):
        with pytest.raises(ValueError):
            BenchmarkConfig(model=ModelConfig(name="graphgps"), model_options=options).validate()
    with pytest.raises(ValueError):
        BenchmarkConfig(
            model=ModelConfig(name="sgformer"), model_options={"use_weight": "yes"}
        ).validate()


def test_recommendations_require_primary_complete_comparable_measurements():
    from efficientgraphbench.results.recommend import recommend

    common = dict(
        schema_version=4,
        status="SUCCESS",
        category="primary",
        source_type="official_repo",
        comparison_group="g",
        model="graphgps",
        accuracy=0.85,
        inference_latency_median_ms=3,
        peak_gpu_reserved_mib=100,
        seed=42,
    )
    rows = [common, {**common, "seed": 43}]
    assert len(recommend(rows, 0.8, 10, 200)) == 1
    assert not recommend(rows, 0.9, 10, 200)
    assert not recommend(rows, 0.8, 10, 50)
    assert not recommend([{**r, "category": "experimental"} for r in rows], 0.8, 10, 200)
    partial = [common, {**common, "status": "OOM", "seed": 43}]
    assert not recommend(partial, 0.8, 10, 200)
    with pytest.raises(ValueError, match="Multiple"):
        recommend([common, {**common, "comparison_group": "other-hardware"}], 0.8, 10, 200)


def test_scaling_retains_masks_and_isolates_size_outputs(tmp_path, monkeypatch):
    from efficientgraphbench.benchmark.scaling import run_scaling
    from efficientgraphbench.datasets.custom import load_custom

    captured = []
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.scaling.load_dataset", lambda *a, **k: bundle()
    )

    def capture(config):
        graph, _, _ = load_custom(config.dataset_path)
        captured.append((graph, config.output_dir, config._scaling_context))
        return {"status": "SUCCESS"}

    monkeypatch.setattr("efficientgraphbench.benchmark.scaling.run_benchmark", capture)
    path = run_scaling(BenchmarkConfig(output_dir=str(tmp_path)), [6], ["gcn"], [42])
    assert path.exists()
    graph, output, context = captured[0]
    assert torch.equal(graph.train_mask, bundle().data.train_mask)
    assert context["node_count"] == 6 and context["edge_count"] == 4
    assert output.endswith("nodes-6")
    with pytest.raises(ValueError, match="Sizes"):
        run_scaling(BenchmarkConfig(output_dir=str(tmp_path)), [100], ["gcn"], [42])


def test_early_stopping_records_actual_budget(tmp_path, monkeypatch):
    from efficientgraphbench.training.trainer import train

    graph = bundle().data
    model = build_model(ModelConfig(), 6, 2)
    monkeypatch.setattr("efficientgraphbench.training.trainer.accuracy", lambda *a: 0.5)
    result = train(
        model,
        graph,
        TrainingConfig(epochs=10, early_stopping_patience=2),
        torch.device("cpu"),
        tmp_path / "model.pt",
    )
    assert result["epochs_run"] == 3
    assert result["maximum_epochs"] == 10
    assert result["training_termination_reason"] == "early_stopping"


def test_appnp_uses_pinned_reference_class_and_propagation():
    from torch_geometric.nn import APPNP

    model = build_model(ModelConfig(name="appnp"), 6, 2)
    assert type(model.model).__module__ == "efficientgraphbench_reference_appnp"
    assert isinstance(model.model.prop1, APPNP)
    assert model.model.prop1.K == 10 and model.model.prop1.alpha == 0.1
    assert model.data_interface
    model.eval()
    logits = model(bundle().data)
    torch.testing.assert_close(logits.logsumexp(dim=1), torch.zeros(6), atol=1e-6, rtol=0)


def test_appnp_source_tampering_is_dependency_error(tmp_path, monkeypatch):
    path = tmp_path / "appnp.py"
    path.write_text("class Net: pass")
    monkeypatch.setattr("efficientgraphbench.models.reference_appnp.SOURCE", path)
    with pytest.raises(ImportError, match="differs"):
        build_model(ModelConfig(name="appnp"), 6, 2)


def test_worker_refuses_to_overwrite_existing_exchange(tmp_path):
    path = tmp_path / "request.json"
    path.write_text('"original"')
    with pytest.raises(RunnerError, match="reuse"):
        ExternalEnvironmentRunner([sys.executable], tmp_path).run({})
    assert path.read_text() == '"original"'


def test_worker_timeout_kills_descendants(tmp_path):
    import psutil

    script = tmp_path / "tree.py"
    pid_file = tmp_path / "child.pid"
    script.write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        f"Path({str(pid_file)!r}).write_text(str(p.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    with pytest.raises(RunnerError, match="timeout"):
        ExternalEnvironmentRunner([sys.executable, str(script)], tmp_path / "run", 2).run({})
    assert pid_file.exists()
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_scaling_stops_only_oom_model_and_keeps_last_success(tmp_path, monkeypatch):
    from efficientgraphbench.benchmark.scaling import run_scaling

    original = bundle()
    original.data.x = torch.eye(18)
    original.data.y = torch.arange(18) % 2
    for split in ("train", "val", "test"):
        mask = getattr(original.data, split + "_mask")
        setattr(
            original.data, split + "_mask", torch.cat([mask, torch.zeros(12, dtype=torch.bool)])
        )
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.scaling.load_dataset", lambda *a, **k: original
    )
    calls = []

    def run(cfg):
        size = cfg._scaling_context["node_count"]
        calls.append((cfg.model.name, size, cfg.seed))
        return {
            "status": "OOM" if cfg.model.name == "gat" and size == 12 else "SUCCESS",
            "accuracy": 0.75,
        }

    monkeypatch.setattr("efficientgraphbench.benchmark.scaling.run_benchmark", run)
    run_scaling(BenchmarkConfig(output_dir=str(tmp_path)), [], ["gat", "gcn"], [42, 43, 44], step=6)
    assert ("gat", 12, 42) in calls
    assert ("gat", 12, 43) not in calls
    assert not any(model == "gat" and size == 18 for model, size, seed in calls)
    assert ("gcn", 18, 44) in calls
    summary = json.loads((tmp_path / "scaling-summary.json").read_text())
    assert summary["models"]["gat"]["last_success"]["num_nodes"] == 6
    assert summary["models"]["gat"]["first_oom"]["num_nodes"] == 12
    assert len(summary["skipped_runs"]) == 5


def test_matrix_skips_remaining_seeds_after_oom(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from efficientgraphbench.cli import main

    calls = []

    def run(cfg):
        calls.append((cfg.model.name, cfg.seed))
        return {"status": "OOM" if cfg.model.name == "gat" else "SUCCESS"}

    monkeypatch.setattr(main, "run_benchmark", run)
    monkeypatch.setattr(main, "compare", lambda *a: None)
    monkeypatch.setattr(main, "write_report", lambda *a: "report")
    result = CliRunner().invoke(
        main.app, ["matrix", "--models", "gat,gcn", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert calls == [("gat", 42), ("gcn", 42), ("gcn", 43), ("gcn", 44)]
    skipped = json.loads((tmp_path / "matrix-skipped.json").read_text())
    assert len(skipped["skipped_runs"]) == 2
