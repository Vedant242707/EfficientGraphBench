import copy

import pytest
import torch
from torch_geometric.data import Data

from efficientgraphbench import Benchmark, ProgressiveBenchmark
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.progressive.checkpoints import CheckpointManager
from efficientgraphbench.progressive.config import FidelityStage, ProgressiveConfig
from efficientgraphbench.progressive.graphs import ProgressionBuilder
from efficientgraphbench.progressive.selection import select_candidates


def bundle():
    n = 60
    generator = torch.Generator().manual_seed(2)
    ids = torch.arange(n)
    data = Data(
        x=torch.randn(n, 4, generator=generator),
        y=ids % 3,
        edge_index=torch.stack([ids, (ids + 1) % n]),
        train_mask=ids % 3 == 0,
        val_mask=ids % 3 == 1,
        test_mask=ids % 3 == 2,
    )
    return DatasetBundle(data, 3)


def config(**kw):
    return ProgressiveConfig(
        stages=[FidelityStage("initial", 0.5, 2), FidelityStage("full", 1.0, 2)], **kw
    ).validate()


def test_nested_deterministic_and_masks(tmp_path):
    first = ProgressionBuilder(bundle(), config(), tmp_path / "first")
    a, b = first.build(0), first.build(1)
    assert set(a.node_ids.tolist()) < set(b.node_ids.tolist())
    assert torch.equal(a.data.val_mask, b.data.val_mask[:30])
    assert not (b.data.train_mask & b.data.val_mask).any()
    assert b.old_mask.sum() == 30
    second = ProgressionBuilder(bundle(), config(), tmp_path / "second")
    assert second.build(0).metadata["stage_graph_hash"] == a.metadata["stage_graph_hash"]
    different = ProgressionBuilder(bundle(), config(progression_seed=4), tmp_path / "different")
    assert different.build(0).node_ids.tolist() != a.node_ids.tolist()
    # No validation or test label stratification: changing future labels does not change order.
    changed = bundle()
    changed.data.y[~changed.data.train_mask] = 0
    assert torch.equal(
        ProgressionBuilder(changed, config(), tmp_path / "labels").order, first.order
    )


def test_temporal_no_future_edges(tmp_path):
    data = bundle()
    data.data.node_time = torch.arange(60)
    data.data.edge_time = torch.maximum(data.data.edge_index[0], data.data.edge_index[1])
    data.data.edge_time[0] = 59  # Edge between early nodes does not exist yet.
    with pytest.raises(ValueError, match="Temporal progression requires"):
        ProgressionBuilder(data, config(), tmp_path / "unsafe")
    builder = ProgressionBuilder(data, config(temporal_static_features=True), tmp_path / "safe")
    early, full = builder.build(0), builder.build(1)
    assert early.metadata["progression_strategy"] == "temporal"
    assert not ((early.data.edge_index[0] == 0) & (early.data.edge_index[1] == 1)).any()
    assert full.data.num_edges == data.data.num_edges
    assert early.node_ids.tolist() == list(range(30))


@pytest.mark.parametrize(
    "stages",
    [
        [],
        [FidelityStage("x", 0.5, 2)],
        [FidelityStage("x", 0.8, 2), FidelityStage("x", 1, 2)],
        [FidelityStage("x", 0.8, 2), FidelityStage("y", 0.7, 2), FidelityStage("z", 1, 2)],
        [FidelityStage("x", 1, 0)],
    ],
)
def test_invalid_stage_configs(stages):
    with pytest.raises(ValueError):
        ProgressiveConfig(stages=stages).validate()


def test_promotion_validation_only_and_safety():
    stage = FidelityStage("s", 0.5, 2, retain_fraction=0.1)
    records = [
        dict(model=m, status="SUCCESS", validation_metric=s, test_accuracy=t)
        for m, s, t in [("gcn", 0.8, 0.1), ("mlp", 0.8, 0.99), ("gat", 0.7, 1.0)]
    ]
    decisions = select_candidates(records, stage, 2)
    assert [m for m, d in decisions.items() if d["candidate_status"] == "advanced"] == [
        "gcn",
        "mlp",
    ]
    records[0]["status"] = "OOM"
    assert "gcn" not in select_candidates(records, stage, 2)


@pytest.mark.parametrize(
    "key",
    [
        "seed",
        "model",
        "model_config_hash",
        "stage_graph_hash",
        "stage_index",
        "software_provenance",
    ],
)
def test_checkpoint_rejects_mismatch(tmp_path, key):
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters())
    identity = dict(
        seed=1,
        model="mlp",
        model_config_hash="config",
        stage_graph_hash="graph",
        stage_index=0,
        software_provenance="software",
    )
    path = tmp_path / "checkpoint.pt"
    digest = CheckpointManager.save(path, model, optimizer, None, identity)
    with pytest.raises(ValueError, match=key):
        CheckpointManager.restore(path, digest, {**identity, key: "wrong"}, model, optimizer, None)


def test_optimizer_and_rng_continuation(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    def step():
        optimizer.zero_grad()
        model(torch.randn(5, 4)).square().sum().backward()
        optimizer.step()
        scheduler.step()

    step()
    path = tmp_path / "checkpoint.pt"
    digest = CheckpointManager.save(path, model, optimizer, scheduler, {})
    step()
    expected = copy.deepcopy(model.state_dict())
    CheckpointManager.restore(path, digest, {}, model, optimizer, scheduler)
    step()
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        CheckpointManager.restore(path, digest, {}, model, optimizer, scheduler)


def test_end_to_end_lineage_and_separate_reporting(tmp_path):
    results = ProgressiveBenchmark(
        bundle(),
        models=["gcn", "mlp", "graphormer"],
        seeds=[1, 2],
        progressive=config(),
        device="cpu",
        output_dir=str(tmp_path),
    ).run()
    assert (results.output_dir / "progressive-report.html").exists()
    assert len(results.records) == 10
    successful = [r for r in results.records if r["status"] == "SUCCESS"]
    for row in successful:
        assert row["checkpoint_policy"]["optimizer_state"] == "resume"
        if row["stage_index"] == 1:
            parent = next(
                r
                for r in successful
                if r["stage_index"] == 0 and r["model"] == row["model"] and r["seed"] == row["seed"]
            )
            assert row["parent_checkpoint_id"] == parent["checkpoint_id"]
            assert row["current_epoch"] == 4
            adaptation = row["adaptation"]
            assert adaptation["old_node_performance_delta"] == pytest.approx(
                adaptation["old_node_metric"] - parent["validation_metric"]
            )
            assert adaptation["post_expansion_delta"] == pytest.approx(
                adaptation["old_node_score_pre_training"] - parent["validation_metric"]
            )
            assert row["adaptation"]["old_node_count"] + row["adaptation"]["new_node_count"] == 20
            assert row["test_metrics"] is not None
        else:
            assert row["test_metrics"] is None
    assert results.metadata["compute"]["wall_clock_reduction_percent"] is None
    unsupported = [r for r in results.records if r["model"] == "graphormer"]
    assert all(r["failure_reason"] == "progressive_resume_unsupported" for r in unsupported)


def test_full_mode_dispatch_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(
        "efficientgraphbench.api.benchmark.run_matrix", lambda jobs: called.extend(jobs) or []
    )
    Benchmark("texas", models=["gcn"], seeds=[2]).run()
    assert len(called) == 1 and called[0].seed == 2


def test_failure_does_not_advance(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr("efficientgraphbench.progressive.controller.train_stage", fail)
    results = ProgressiveBenchmark(
        bundle(),
        models=["gcn"],
        seeds=[1, 2],
        progressive=config(),
        device="cpu",
        output_dir=str(tmp_path),
    ).run()
    assert [r["status"] for r in results.records] == ["OOM", "SKIPPED_DETERMINISTIC_FAILURE"]
    assert all(r["fidelity"]["stage_index"] == 0 for r in results.records)


@pytest.mark.parametrize("model", ["graphsage", "gat", "gatv2", "sgc", "appnp", "sgformer_pyg"])
def test_verified_adapter_growth(tmp_path, model):
    results = ProgressiveBenchmark(
        bundle(),
        models=[model],
        seeds=[1],
        progressive=config(),
        device="cpu",
        output_dir=str(tmp_path),
    ).run()
    assert [r["status"] for r in results.records] == ["SUCCESS", "SUCCESS"], results.records
    assert results.records[1]["parent_checkpoint_id"] == results.records[0]["checkpoint_id"]


def test_progressive_api_dispatch(tmp_path):
    results = Benchmark(
        bundle(),
        models=["mlp"],
        seeds=[1],
        benchmark_mode="progressive_multi_fidelity",
        progressive=config(),
        device="cpu",
        output_dir=str(tmp_path),
    ).run()
    assert len(results.records) == 2
    assert all(r["benchmark_mode"] == "progressive_multi_fidelity" for r in results.records)
