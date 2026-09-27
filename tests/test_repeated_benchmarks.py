"""Offline protocol checks: never download or train large datasets."""

import torch
from typer.testing import CliRunner

from efficientgraphbench import Benchmark, BenchmarkResults
from efficientgraphbench.cli.main import app
from efficientgraphbench.config import BenchmarkConfig
from efficientgraphbench.datasets.validation import stratified_masks


def test_repeat_jobs_share_splits_between_models(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        "efficientgraphbench.api.benchmark.run_matrix", lambda jobs: captured.extend(jobs) or []
    )
    Benchmark(
        dataset="roman-empire",
        models=["gcn", "gat"],
        runs=20,
        splits="official",
        output_dir=str(tmp_path),
    ).run()
    assert len(captured) == 40
    assert [c.seed for c in captured[:20]] == list(range(1, 21))
    assert [c.split_index for c in captured[:20]] == list(range(10)) * 2
    assert [c.split_index for c in captured[20:]] == list(range(10)) * 2
    for config in captured:
        config.validate()


def test_random_repeats_real_training_and_saved_metrics(tmp_path):
    results = Benchmark(
        dataset="custom",
        dataset_path="examples/custom_graph",
        models=["gcn"],
        runs=10,
        splits="random",
        epochs=1,
        device="cpu",
        output_dir=str(tmp_path),
        latency_warmup=0,
        latency_repeats=1,
    ).run()
    assert all(r.status == "SUCCESS" for r in results)
    assert len({r.split_hash for r in results}) > 1
    summary = results.summary()
    assert len(summary) == 1
    assert summary.iloc[0]["runs"] == 10
    assert len(results.summary(across_splits=False)) > 1
    for result in results:
        assert result.config["split_seed"] == result.seed
        assert result.checkpoint_write_time_sec >= 0
        assert result.optimizer_tensor_memory_mib > 0
        assert result.parameter_memory_mib > 0
        assert result.memory_snapshots["training_checkpoint_and_test"]["gpu_allocated_mib"] is None
        assert result.phase_times_sec["inference_warmup_and_measurement"] >= 0
    # Aggregating split variation must not aggregate different hardware.
    changed = dict(results.records[0], hardware_fingerprint="another-device")
    assert len(BenchmarkResults([results.records[0], changed]).summary(across_splits=True)) == 2


def test_graph_cache_reuses_fixed_splits_and_repairs_corruption(tmp_path):
    from pathlib import Path

    from efficientgraphbench.benchmark.interchange import export_graph
    from efficientgraphbench.datasets import load_dataset

    bundle = load_dataset("custom", dataset_path="examples/custom_graph")
    first = export_graph(bundle, tmp_path / "first", cache_dir=tmp_path / "cache")
    second = export_graph(bundle, tmp_path / "second", cache_dir=tmp_path / "cache")
    assert first["graph_path"] == second["graph_path"]
    Path(first["graph_path"]).write_bytes(b"corrupt")
    repaired = export_graph(bundle, tmp_path / "third", cache_dir=tmp_path / "cache")
    assert repaired["graph_file_sha256"] == first["graph_file_sha256"]


def test_random_splits_exclude_unlabeled_nodes():
    labels = torch.tensor([-1, 0, 0, 0, 1, 1, 1])
    for mask in stratified_masks(labels, 1):
        assert not mask[0]
        assert mask.sum() == 2


def test_official_fold_selects_all_three_masks(monkeypatch, tmp_path):
    from efficientgraphbench.datasets import load_dataset

    graph = load_dataset("custom", dataset_path="examples/custom_graph").data
    masks = [stratified_masks(graph.y, seed) for seed in range(10)]
    for position, key in enumerate(("train_mask", "val_mask", "test_mask")):
        setattr(graph, key, torch.stack([fold[position] for fold in masks], dim=1))

    class ActorFixture:
        num_classes = 2

        def __init__(self, *args):
            pass

        def __getitem__(self, index):
            return graph.clone()

    monkeypatch.setattr("efficientgraphbench.datasets.registry.Actor", ActorFixture)
    loaded = load_dataset("actor", root=tmp_path, split_index=9)
    for position, key in enumerate(("train_mask", "val_mask", "test_mask")):
        assert torch.equal(getattr(loaded.data, key), masks[9][position])
    assert loaded.split == "actor-official-9"


def test_repeat_cli_help_and_invalid_runs():
    runner = CliRunner()
    assert runner.invoke(app, ["repeat", "--help"]).exit_code == 0
    result = runner.invoke(app, ["repeat", "--runs", "0"])
    assert result.exit_code == 1
    assert "positive integer" in result.output


def test_ogb_default_keeps_official_split(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "efficientgraphbench.api.benchmark.run_matrix", lambda jobs: captured.extend(jobs) or []
    )
    Benchmark(dataset="ogbn-products", models="gcn", runs=20).run()
    assert len(captured) == 20
    assert all(c.split_mode == "official" and c.split_index == 0 for c in captured)
    BenchmarkConfig(dataset="ogbn-products").validate()
