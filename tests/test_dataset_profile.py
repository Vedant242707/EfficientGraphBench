import json

import pytest
import torch
from torch_geometric.data import Data
from typer.testing import CliRunner

from efficientgraphbench.cli.main import app
from efficientgraphbench.datasets import DatasetBundle, profile_dataset


def test_profile_exact_statistics_and_no_mutation(tmp_path):
    # 0--1--2, plus isolate 3; duplicate arc and self-loop must not bias metrics.
    data = Data(
        x=torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 1, 2]]),
        y=torch.tensor([0, 0, 1, -1]),
    )
    old_edges = data.edge_index.clone()
    p = DatasetBundle(data, 2).profile()
    assert p["num_edges"] == 6
    assert p["unique_nonself_arcs"] == 4
    assert p["avg_degree"] == 1
    assert p["median_degree"] == 1
    assert p["max_degree"] == 2
    assert p["graph_density"] == pytest.approx(1 / 3)
    assert p["edge_homophily"] == 0.5
    assert p["connected_components"] == 2
    assert p["largest_component_pct"] == 75
    assert p["isolated_nodes"] == 1
    assert p["feature_sparsity_pct"] == 75
    assert p["class_counts"] == [2, 1]
    assert p["class_imbalance"] == "moderate"
    assert torch.equal(data.edge_index, old_edges)
    path = p.save_json(tmp_path / "profile.json")
    assert json.loads(path.read_text())["unlabeled_nodes"] == 1


def test_directed_and_unlabeled_edges():
    data = Data(
        x=torch.ones(3, 2), y=torch.tensor([0, -1, 1]), edge_index=torch.tensor([[0, 1], [1, 2]])
    )
    p = profile_dataset(DatasetBundle(data, 2))
    assert not p["is_undirected"]
    assert p["degree_definition"] == "unique outgoing neighbors"
    assert p["connected_components"] == 1
    assert p["edge_homophily"] is None
    assert p["feature_sparsity_pct"] == 0


def test_single_node_empty_edges_and_raw_features():
    data = Data(
        x=torch.ones(1, 2),
        raw_x=torch.zeros(1, 2),
        y=torch.tensor([0]),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    p = profile_dataset(DatasetBundle(data, 2))
    assert p["graph_density"] == 0
    assert p["largest_component_pct"] == 100
    assert p["edge_homophily"] is None
    assert p["class_imbalance_ratio"] is None
    assert p["class_imbalance"] == "high"
    assert p["feature_sparsity_pct"] == 100
    assert p["feature_source"] == "raw_x"


def test_profile_cli(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "profile",
            "--dataset",
            "custom",
            "--dataset-path",
            "examples/custom_graph",
            "--output",
            str(tmp_path / "p.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Homophily" in result.output
    assert "Feature sparsity" in result.output
    assert json.loads((tmp_path / "p.json").read_text())["num_nodes"] == 12
