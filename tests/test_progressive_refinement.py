import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from efficientgraphbench.api.results import BenchmarkResults
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.progressive.config import FidelityStage, ProgressiveConfig
from efficientgraphbench.progressive.controller import ProgressiveBenchmark
from efficientgraphbench.progressive.results import recovery_summary
from efficientgraphbench.progressive.selection import select_candidates


def test_protocol_isolation_and_legacy():
    full = dict(model="gcn", status="SUCCESS", accuracy=0.5)
    BenchmarkResults([full]).summary()
    for extra in (
        {"benchmark_mode": "progressive_multi_fidelity"},
        {"fidelity": {}},
        {"adaptation": {}},
    ):
        with pytest.raises(ValueError, match="Progressive"):
            BenchmarkResults([full, {**full, **extra}])
    result = BenchmarkResults([full])
    result.records.append({**full, "benchmark_mode": "progressive_multi_fidelity"})
    with pytest.raises(ValueError):
        result.summary()


def test_expected_seed_coverage():
    stage = FidelityStage("first", 0.5, 1)
    row = dict(model="gcn", status="SUCCESS", validation_metric=0.8, seed=1)
    assert not select_candidates([row], stage, 1, expected_seeds=[1, 2])
    assert not select_candidates([row, row], stage, 1, expected_seeds=[1, 2])
    assert select_candidates([row, {**row, "seed": 2}], stage, 1, expected_seeds=[1, 2])


def test_progressive_rejects_full_and_mutated_records(tmp_path):
    from efficientgraphbench.progressive.results import ProgressiveResults

    with pytest.raises(ValueError, match="mixed protocols"):
        ProgressiveResults([dict(model="gcn", status="SUCCESS")], tmp_path, {})
    result = ProgressiveResults([], tmp_path, {})
    result.records.append(dict(model="gcn", benchmark_mode="full", fidelity={}))
    with pytest.raises(ValueError, match="mixed protocols"):
        result.summary()


def test_recovery_summary_relapse():
    rows = [
        {
            "adaptation": {
                "previous_stage_score": 0.8,
                "recovery_status": state,
                "final_target_satisfied": final,
            }
        }
        for state, final in [
            ("recovery_not_required", True),
            ("recovered", False),
            ("recovered", True),
            ("not_recovered", False),
        ]
    ]
    summary = recovery_summary(rows)
    assert summary["Transitions requiring recovery"] == 3
    assert summary["Recovered transitions"] == 2
    assert summary["Never recovered"] == 1
    assert summary["Final target satisfied"] == 2


@pytest.mark.parametrize(
    "before,scores,final,status,epoch",
    [
        (0.8, [0.2, 0.3], 0.3, "recovery_not_required", 0),
        (0.2, [0.3, 0.8], 0.8, "recovered", 2),
        (0.2, [0.3, 0.8, 0.4], 0.4, "recovered", 2),
        (0.2, [0.3, 0.4], 0.4, "not_recovered", None),
    ],
)
def test_first_recovery_and_final_relapse(before, scores, final, status, epoch):
    from efficientgraphbench.progressive.adaptation import RecoveryTracker

    tracker = RecoveryTracker(0.8, before, 0.95)
    for i, score in enumerate(scores, 1):
        tracker.observe(i, score, float(i))
    result = tracker.result(final)
    assert result["recovery_status"] == status
    assert result["first_recovery_epoch"] == epoch
    assert result["final_target_satisfied"] == (final >= 0.76)


def test_failed_time_preserved(monkeypatch):
    from efficientgraphbench.progressive import training

    def fail(*args, timings, **kwargs):
        timings["optimization_seconds"] = 0.1
        raise RuntimeError("failure during training")

    monkeypatch.setattr(training, "_train_stage", fail)
    with pytest.raises(RuntimeError) as caught:
        training.train_stage()
    assert caught.value.progressive_measurements["optimization_seconds"] == 0.1
    assert caught.value.progressive_measurements["failed_attempt_elapsed_seconds"] >= 0


def test_fingerprint_ignores_presentation():
    from efficientgraphbench.benchmark.execution_identity import execution_sources

    sources = execution_sources()
    assert "training/trainer.py" in sources
    assert "runners/reference_worker.py" in sources
    assert not any(k.startswith("results/") or k == "progressive/results.py" for k in sources)


@pytest.mark.parametrize("model", ["sgformer", "graphgps"])
def test_official_reference_growth(tmp_path, model):
    from efficientgraphbench.models.catalog import model_spec

    spec = model_spec(model)
    if not Path(spec["python"]).exists() or not Path(spec["repository"]).exists():
        pytest.skip("Actual pinned reference environment is not installed")
    ids = torch.arange(40)
    gen = torch.Generator().manual_seed(51)
    graph = Data(
        x=torch.randn(40, 6, generator=gen),
        y=ids % 2,
        edge_index=torch.stack(
            [torch.cat([ids, (ids + 1) % 40]), torch.cat([(ids + 1) % 40, ids])]
        ),
        train_mask=ids % 5 < 3,
        val_mask=ids % 5 == 3,
        test_mask=ids % 5 == 4,
    )
    results = ProgressiveBenchmark(
        DatasetBundle(graph, 2),
        models=[model],
        seeds=[1],
        progressive=ProgressiveConfig(
            stages=[FidelityStage("initial", 0.5, 2), FidelityStage("full", 1, 2)]
        ),
        device="cpu",
        output_dir=str(tmp_path),
    ).run()
    assert all(r["status"] == "SUCCESS" for r in results.records), json.dumps(
        results.records, default=str
    )[-6000:]
    first, last = results.records
    saved = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=False)
    final = torch.load(last["checkpoint_path"], map_location="cpu", weights_only=False)
    assert last["parent_checkpoint_id"] == first["checkpoint_id"]
    assert last["resume_evidence"]["saved_model_signature"] == saved["saved_model_signature"]
    assert (
        last["resume_evidence"]["saved_optimizer_signature"] == saved["saved_optimizer_signature"]
    )
    assert final["current_epoch"] == 4
    assert final["software_provenance"]["commit"] == spec["commit"]
    assert all(int(state["step"]) == 4 for state in final["optimizer_state_dict"]["state"].values())
    assert first["test_metrics"] is None and last["test_metrics"] is not None
    assert last["adaptation"]["new_node_score_pre_training"] is not None
    assert (
        last["reference_settings"]["preprocessing_stage_graph_hash"]
        != first["reference_settings"]["preprocessing_stage_graph_hash"]
    )
    expected_class = (
        "graphgps.network.gps_model.GPSModel" if model == "graphgps" else "ours.SGFormer"
    )
    assert last["reference_settings"]["implementation_class"] == expected_class
