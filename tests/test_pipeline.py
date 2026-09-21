import json

import pytest
import torch
from torch_geometric.data import Data
from typer.testing import CliRunner

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.cli.main import app
from efficientgraphbench.config import BenchmarkConfig, ModelConfig, TrainingConfig, load_config
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.models.registry import build_model
from efficientgraphbench.results.writer import read_results, write_result
from efficientgraphbench.training.evaluator import accuracy
from efficientgraphbench.training.seed import seed_everything


@pytest.fixture
def graph():
    seed_everything(7)
    n = 24
    source = torch.arange(n)
    edge_index = torch.stack(
        [torch.cat([source, source]), torch.cat([(source + 1) % n, (source - 1) % n])]
    )
    return Data(
        x=torch.rand(n, 8),
        edge_index=edge_index,
        y=source % 3,
        train_mask=source < 12,
        val_mask=(source >= 12) & (source < 18),
        test_mask=source >= 18,
    )


@pytest.mark.parametrize(
    "name",
    [
        "gcn",
        "graphsage",
        "gat",
        "sgformer_pyg",
        "mlp",
        "sgc",
        "appnp",
        "gatv2",
        "graphormer_adapted",
        "graphgps_adapted",
    ],
)
def test_end_to_end(tmp_path, monkeypatch, graph, name):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset",
        lambda *args, **kwargs: DatasetBundle(graph.clone(), 3, split="synthetic-test"),
    )
    cfg = BenchmarkConfig(
        model=ModelConfig(name=name, hidden_dim=8),
        device="cpu",
        training=TrainingConfig(epochs=2, latency_warmup=1, latency_repeats=2),
        output_dir=str(tmp_path),
    )
    result = run_benchmark(cfg)
    assert result["status"] == "SUCCESS"
    assert 0 <= result["accuracy"] <= 1
    assert result["num_nodes"] == 24
    assert result["peak_cpu_memory_mb"] > 0
    assert result["peak_gpu_memory_mb"] is None
    assert result["inference_ms"] > 0
    assert result["train_time_sec"] >= result["optimization_time_sec"]
    assert read_results(tmp_path)[0]["run_id"] == result["run_id"]
    assert (tmp_path / "raw" / "results.csv").exists()
    checkpoint = torch.load(result["checkpoint_path"], weights_only=True)
    model = build_model(cfg.model, 8, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    assert accuracy(model(graph), graph.y, graph.test_mask) == result["accuracy"]


def test_seed_reproducibility(graph):
    outputs = []
    for _ in range(2):
        seed_everything(42)
        model = build_model(ModelConfig(), 8, 3)
        outputs.append(model(graph))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


def test_training_reproducibility(tmp_path, monkeypatch, graph):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset",
        lambda *args, **kwargs: DatasetBundle(graph.clone(), 3, split="synthetic-test"),
    )
    cfg = BenchmarkConfig(
        device="cpu",
        output_dir=str(tmp_path),
        training=TrainingConfig(epochs=3, latency_warmup=0, latency_repeats=1),
    )
    first, second = run_benchmark(cfg), run_benchmark(cfg)
    assert first["accuracy"] == second["accuracy"]
    assert first["best_epoch"] == second["best_epoch"]
    a = torch.load(first["checkpoint_path"], weights_only=True)["model_state_dict"]
    b = torch.load(second["checkpoint_path"], weights_only=True)["model_state_dict"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_oom_is_result(tmp_path, monkeypatch, graph):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset",
        lambda *args, **kwargs: DatasetBundle(graph, 3),
    )

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated allocation failure")

    monkeypatch.setattr("efficientgraphbench.benchmark.runner.build_model", fail)
    result = run_benchmark(BenchmarkConfig(device="cpu", output_dir=str(tmp_path)))
    assert result["status"] == "OOM"
    assert "accuracy" not in result
    assert read_results(tmp_path)[0]["status"] == "OOM"


def test_failure_is_saved(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("dataset download failed")

    monkeypatch.setattr("efficientgraphbench.benchmark.runner.load_dataset", fail)
    result = run_benchmark(BenchmarkConfig(device="cpu", output_dir=str(tmp_path)))
    assert result["status"] == "PREPROCESSING_ERROR"
    assert read_results(tmp_path)[0]["status"] == "PREPROCESSING_ERROR"


def test_invalid_config_and_empty_accuracy(tmp_path):
    with pytest.raises(ValueError):
        BenchmarkConfig(training=TrainingConfig(epochs=0)).validate()
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true")
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        accuracy(torch.zeros(2, 2), torch.zeros(2), torch.zeros(2, dtype=torch.bool))


def test_corrupt_storage_is_not_overwritten(tmp_path):
    write_result({"run_id": "first", "status": "ok"}, tmp_path)
    path = tmp_path / "raw" / "results.jsonl"
    path.write_text(path.read_text() + "broken\n")
    with pytest.raises(ValueError, match="corrupt result"):
        write_result({"run_id": "second"}, tmp_path)
    assert json.loads(path.read_text().splitlines()[0])["run_id"] == "first"


def test_cli_help_and_validation(tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["run", "--epochs", "0", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "epochs" in result.output


def test_public_names_and_pubmed_matrix_config(tmp_path, monkeypatch):
    jobs = []

    def fake_run(config):
        jobs.append(config)
        return {"status": "ok"}

    monkeypatch.setattr("efficientgraphbench.cli.main.run_benchmark", fake_run)
    monkeypatch.setattr("efficientgraphbench.cli.main.compare", lambda *args: None)
    monkeypatch.setattr("efficientgraphbench.cli.main.write_report", lambda *args: "report.md")
    config = tmp_path / "pubmed.yaml"
    config.write_text("dataset: PubMed\nmodel:\n  name: GraphSAGE\n")
    result = CliRunner().invoke(app, ["matrix", "--config", str(config), "--seeds", "42"])
    assert result.exit_code == 0, result.output
    assert len(jobs) == 10
    assert all(job.dataset == "pubmed" for job in jobs)
    assert BenchmarkConfig(model=ModelConfig(name="SGFormer")).validate().model.name == "sgformer"


def test_display_names_in_comparison_and_report(tmp_path):
    record = {
        "run_id": "names-test",
        "dataset": "pubmed",
        "model": "graphsage",
        "status": "ok",
        "comparison_group": "test",
        "device_name": "Test CPU",
        "device": "cpu",
        "torch_version": "test",
        "pyg_version": "test",
        "seed": 42,
        "config": BenchmarkConfig().to_dict(),
        "accuracy": 0.8,
        "train_time_sec": 1.0,
        "inference_ms": 2.0,
        "peak_gpu_memory_mb": None,
        "peak_cpu_memory_mb": 100.0,
    }
    write_result(record, tmp_path)
    result = CliRunner().invoke(
        app, ["compare", "--dataset", "PubMed", "--output-dir", str(tmp_path), "--report"]
    )
    assert result.exit_code == 0, result.output
    assert "GraphSAGE" in result.output
    report = (tmp_path / "reports" / "pubmed.md").read_text(encoding="utf-8")
    assert "# PubMed benchmark report" in report
    assert "| GraphSAGE |" in report


def test_mlp_ignores_connections(graph):
    model = build_model(ModelConfig(name="mlp", dropout=0), 8, 3).eval()
    changed = graph.clone()
    changed.edge_index = torch.empty((2, 0), dtype=torch.long)
    assert torch.equal(model(graph), model(changed))


def test_gatv2_uses_dynamic_attention():
    from torch_geometric.nn import GATv2Conv

    model = build_model(ModelConfig(name="gatv2"), 8, 3)
    assert all(isinstance(conv, GATv2Conv) for conv in model.model.convs)


@pytest.mark.parametrize("name", ["sgc", "appnp"])
def test_propagation_does_not_reuse_another_graph(graph, name):
    config = ModelConfig(name=name, dropout=0)
    model = build_model(config, 8, 3).eval()
    model(graph)
    changed = graph.clone()
    changed.x = changed.x * 2
    changed.edge_index = torch.empty((2, 0), dtype=torch.long)
    fresh = build_model(config, 8, 3).eval()
    fresh.load_state_dict(model.state_dict())
    assert torch.allclose(model(changed), fresh(changed))


@pytest.mark.parametrize(
    "kwargs", [{"propagation_steps": 0}, {"alpha": 0}, {"alpha": 1.1}, {"alpha": float("nan")}]
)
def test_invalid_propagation_config(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(model=ModelConfig(**kwargs)).validate()


@pytest.mark.parametrize("name", ["graphormer_adapted", "graphgps_adapted"])
def test_transformer_node_permutation_and_repeated_eval(graph, name):
    model = build_model(ModelConfig(name=name, hidden_dim=8), 8, 3).eval()
    order = torch.randperm(graph.num_nodes)
    inverse = torch.argsort(order)
    permuted = graph.clone()
    permuted.x = graph.x[order]
    permuted.edge_index = inverse[graph.edge_index]
    with torch.no_grad():
        expected = model(graph)
        assert torch.allclose(expected, model(graph), atol=1e-6)
        assert torch.allclose(expected[order], model(permuted), atol=1e-5)
    with pytest.raises(ValueError, match="divisible"):
        BenchmarkConfig(model=ModelConfig(name=name, hidden_dim=7)).validate()


def test_graphormer_structural_encodings_and_limit():
    from efficientgraphbench.models.graphormer import MAX_NODES, prepare_graphormer

    data = Data(x=torch.ones(4, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    prepare_graphormer(data)
    assert data.graphormer_spatial[0].tolist() == [0, 1, 2, 32]
    assert data.graphormer_spatial[2, 0].item() == 32
    assert data.graphormer_in_degree.tolist() == [0, 1, 1, 0]
    assert data.graphormer_out_degree.tolist() == [1, 1, 0, 0]
    from efficientgraphbench.results.status import UnsupportedGraphSize

    with pytest.raises(UnsupportedGraphSize, match="quadratic"):
        prepare_graphormer(
            Data(x=torch.zeros(MAX_NODES + 1, 2), edge_index=torch.empty((2, 0), dtype=torch.long))
        )


def test_graphormer_bias_learns(graph):
    model = build_model(ModelConfig(name="graphormer_adapted", hidden_dim=8, dropout=0), 8, 3)
    torch.nn.functional.cross_entropy(model(graph), graph.y).backward()
    assert model.model.spatial.weight.grad.abs().sum() > 0
    assert model.model.in_degree.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="model must"):
        BenchmarkConfig(model=ModelConfig(name="graphtransformer")).validate()


def test_size_failure_metadata_and_no_preprocessing_allocation(tmp_path, monkeypatch, graph):
    monkeypatch.setattr(
        "efficientgraphbench.benchmark.runner.load_dataset", lambda *a, **k: DatasetBundle(graph, 3)
    )
    monkeypatch.setattr("efficientgraphbench.models.graphormer.MAX_NODES", 20)
    cfg = BenchmarkConfig(
        model=ModelConfig(name="graphormer_adapted"), device="cpu", output_dir=str(tmp_path)
    )
    result = run_benchmark(cfg)
    assert result["status"] == "UNSUPPORTED_GRAPH_SIZE"
    assert result["num_nodes"] == 24 and result["num_edges"] == 48
    assert result["implementation_limit"]["max_nodes"] == 20
    assert result["hardware"]["device"] == "cpu"
    assert result["failure_reason"]
    assert getattr(graph, "graphormer_spatial", None) is None
    from efficientgraphbench.results.report import write_report

    text = write_report("cora", str(tmp_path)).read_text(encoding="utf-8")
    assert "UNSUPPORTED_GRAPH_SIZE" in text
    displayed = CliRunner().invoke(app, ["compare", "--output-dir", str(tmp_path)])
    assert "Failed runs" in displayed.output


def test_config_error_is_saved(tmp_path):
    result = run_benchmark(
        BenchmarkConfig(training=TrainingConfig(epochs=0), output_dir=str(tmp_path))
    )
    assert result["status"] == "CONFIG_ERROR"
    assert result["num_nodes"] is None
    assert "hardware" in result and "failure_reason" in result


def test_matrix_continues_after_failure(tmp_path, monkeypatch):
    jobs = []

    def execute(cfg):
        jobs.append(cfg.model.name)
        return {"status": "UNSUPPORTED_GRAPH_SIZE" if len(jobs) == 1 else "SUCCESS"}

    monkeypatch.setattr("efficientgraphbench.cli.main.run_benchmark", execute)
    monkeypatch.setattr("efficientgraphbench.cli.main.compare", lambda *a: None)
    monkeypatch.setattr("efficientgraphbench.cli.main.write_report", lambda *a: "report.md")
    result = CliRunner().invoke(app, ["matrix", "--seeds", "42", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert len(jobs) == 10
