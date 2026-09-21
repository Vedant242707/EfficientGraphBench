import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from typer.testing import CliRunner

from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.cli.main import app
from efficientgraphbench.config import BenchmarkConfig, ModelConfig, TrainingConfig
from efficientgraphbench.datasets.catalog import DATASETS
from efficientgraphbench.datasets.registry import load_dataset
from efficientgraphbench.datasets.validation import (
    graph_fingerprint,
    stratified_masks,
    validate_graph,
)


@pytest.fixture
def tiny_graph():
    n = 12
    return Data(
        x=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10,
        y=torch.arange(n) % 2,
        edge_index=torch.stack([torch.arange(n), (torch.arange(n) + 1) % n]),
        train_mask=torch.arange(n) < 6,
        val_mask=(torch.arange(n) >= 6) & (torch.arange(n) < 8),
        test_mask=torch.arange(n) >= 8,
    )


def test_generated_splits_fixed_and_disjoint(tiny_graph):
    first = stratified_masks(tiny_graph.y, 10)
    torch.manual_seed(999)
    second = stratified_masks(tiny_graph.y, 10)
    for a, b in zip(first, second):
        assert torch.equal(a, b)
        assert set(tiny_graph.y[a].tolist()) == {0, 1}
    assert torch.all(sum(mask.int() for mask in first) == 1)
    assert not torch.equal(first[0], stratified_masks(tiny_graph.y, 11)[0])


@pytest.mark.parametrize(
    "name", [name for name in DATASETS if name not in {"custom", "ogbn-arxiv"}]
)
def test_pyg_loaders_preserve_protocols(name, tiny_graph, monkeypatch, tmp_path):
    spec = DATASETS[name]
    graph = tiny_graph.clone()
    if spec.loader == "wikics":
        graph.train_mask = torch.stack([graph.train_mask, graph.train_mask], dim=1)
        graph.val_mask = torch.stack([graph.val_mask, graph.val_mask], dim=1)

    class FakeDataset:
        num_classes = 2

        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return graph.clone()

    constructor = {
        "planetoid": "Planetoid",
        "amazon": "Amazon",
        "coauthor": "Coauthor",
        "wikics": "WikiCS",
        "flickr": "Flickr",
    }[spec.loader]
    monkeypatch.setattr(f"efficientgraphbench.datasets.registry.{constructor}", FakeDataset)
    loaded = load_dataset(name, tmp_path, split_index=1 if spec.loader == "wikics" else 0)
    assert loaded.data.train_mask.ndim == 1
    assert len(loaded.fingerprint) == 64
    if spec.split == "generated":
        assert loaded.split == "stratified-per-class-60-20-20-seed0"
    else:
        assert torch.equal(loaded.data.train_mask, tiny_graph.train_mask)
    if spec.loader in {"wikics", "flickr"}:
        torch.testing.assert_close(loaded.data.x, tiny_graph.x)
    else:
        assert loaded.preprocessing == "pyg-normalize-features"


def test_ogb_indices_and_labels(tiny_graph, monkeypatch, tmp_path):
    class FakeOGB:
        num_classes = 2

        def __init__(self, **kwargs):
            assert kwargs["name"] == "ogbn-arxiv"

        def __getitem__(self, index):
            return {
                "node_feat": tiny_graph.x.numpy(),
                "edge_index": tiny_graph.edge_index.numpy(),
            }, tiny_graph.y.numpy()[:, None]

        def get_idx_split(self):
            return {
                key: getattr(tiny_graph, f"{mask}_mask").nonzero().numpy().ravel()
                for key, mask in (("train", "train"), ("valid", "val"), ("test", "test"))
            }

    monkeypatch.setitem(
        sys.modules, "ogb.nodeproppred", SimpleNamespace(NodePropPredDataset=FakeOGB)
    )
    loaded = load_dataset("ogbn-arxiv", tmp_path)
    assert torch.equal(loaded.data.y, tiny_graph.y)
    assert torch.equal(loaded.data.val_mask, tiny_graph.val_mask)
    assert loaded.data.is_undirected()
    assert loaded.split == "ogb-official-time"


@pytest.mark.parametrize("failure", ["overlap", "empty", "nan", "edge", "labels", "missing-class"])
def test_invalid_graph_rejected(tiny_graph, failure):
    if failure == "overlap":
        tiny_graph.val_mask[0] = True
    elif failure == "empty":
        tiny_graph.test_mask[:] = False
    elif failure == "nan":
        tiny_graph.x[0, 0] = float("nan")
    elif failure == "edge":
        tiny_graph.edge_index[0, 0] = 50
    elif failure == "labels":
        tiny_graph.y = tiny_graph.y[:, None]
    else:
        tiny_graph.train_mask &= tiny_graph.y == 0
    with pytest.raises(ValueError):
        validate_graph(tiny_graph, 2)


def test_fingerprint_tracks_graph_and_split(tiny_graph):
    original = graph_fingerprint(tiny_graph)
    copy = tiny_graph.clone()
    copy.x[0, 0] += 1
    assert graph_fingerprint(copy) != original
    copy = tiny_graph.clone()
    copy.train_mask[0] = False
    assert graph_fingerprint(copy) != original
    tiny_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    assert len(graph_fingerprint(tiny_graph)) == 64


def save_npz(path, data):
    np.savez(
        path,
        **{
            key: getattr(data, key).numpy()
            for key in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask")
        },
    )


def test_npz_import_and_bad_class_ids(tiny_graph, tmp_path):
    path = tmp_path / "graph.npz"
    save_npz(path, tiny_graph)
    loaded = load_dataset("custom", tmp_path, dataset_path=path)
    assert loaded.fingerprint == graph_fingerprint(tiny_graph)
    tiny_graph.y[0] = 1000000
    save_npz(path, tiny_graph)
    with pytest.raises(ValueError, match="contiguous"):
        load_dataset("custom", tmp_path, dataset_path=path)


def test_custom_csv_and_validation(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_graph"
    loaded = load_dataset("custom", tmp_path, dataset_path=example)
    assert loaded.label_mapping == {"group-a": 0, "group-b": 1}
    assert loaded.metadata()["num_nodes"] == 12
    (tmp_path / "nodes.csv").write_text((example / "nodes.csv").read_text())
    (tmp_path / "edges.csv").write_text("source,target\nn0,missing\n")
    with pytest.raises(ValueError, match="missing from nodes"):
        load_dataset("custom", tmp_path, dataset_path=tmp_path)


@pytest.mark.parametrize("problem", ["duplicate-node", "bad-feature", "bad-split"])
def test_custom_csv_bad_rows(tmp_path, problem):
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_graph"
    content = (example / "nodes.csv").read_text()
    if problem == "duplicate-node":
        content = content.replace("n1,", "n0,")
    elif problem == "bad-feature":
        content = content.replace("1.0,0.0,0.1", "missing,0.0,0.1")
    else:
        content = content.replace(",train,", ",training,")
    (tmp_path / "nodes.csv").write_text(content)
    (tmp_path / "edges.csv").write_text((example / "edges.csv").read_text())
    with pytest.raises(ValueError):
        load_dataset("custom", tmp_path, dataset_path=tmp_path)


def test_npz_object_arrays_are_rejected(tmp_path):
    path = tmp_path / "objects.npz"
    np.savez(
        path,
        x=np.array([object()], dtype=object),
        edge_index=np.zeros((2, 0)),
        y=np.array([0]),
        train_mask=np.array([True]),
        val_mask=np.array([True]),
        test_mask=np.array([True]),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        load_dataset("custom", tmp_path, dataset_path=path)


def test_custom_graphs_never_merge_comparisons(tiny_graph, tmp_path):
    path = tmp_path / "graph.npz"
    save_npz(path, tiny_graph)
    cfg = BenchmarkConfig(
        dataset="custom",
        dataset_path=str(path),
        device="cpu",
        model=ModelConfig(hidden_dim=8),
        output_dir=str(tmp_path / "out"),
        training=TrainingConfig(epochs=1, latency_warmup=0, latency_repeats=1),
    )
    first = run_benchmark(cfg)
    tiny_graph.x[0, 0] += 0.1
    save_npz(path, tiny_graph)
    second = run_benchmark(cfg)
    assert first["comparison_group"] != second["comparison_group"]
    assert first["dataset_fingerprint"] != second["dataset_fingerprint"]


def test_dataset_list_and_inspection():
    runner = CliRunner()
    listed = runner.invoke(app, ["datasets"])
    assert listed.exit_code == 0
    assert "amazon-photo" in listed.output
    checked = runner.invoke(
        app, ["inspect", "--dataset", "custom", "--dataset-path", "examples/custom_graph"]
    )
    assert checked.exit_code == 0, checked.output
    assert "validation passed" in checked.output
