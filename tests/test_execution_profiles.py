import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from efficientgraphbench import Benchmark
from efficientgraphbench.datasets.registry import DatasetBundle


def small_graph():
    ids = torch.arange(40)
    return DatasetBundle(
        Data(
            x=torch.randn(40, 6, generator=torch.Generator().manual_seed(51)),
            y=ids % 2,
            edge_index=torch.stack([ids, (ids + 1) % 40]),
            train_mask=ids % 5 < 3,
            val_mask=ids % 5 == 3,
            test_mask=ids % 5 == 4,
        ),
        2,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("model", ["gcn", "sgformer", "graphgps"])
def test_real_detailed_trace(tmp_path, device, model):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if model != "gcn":
        from efficientgraphbench.models.catalog import model_spec

        if not Path(model_spec(model)["python"]).exists():
            pytest.skip("Pinned reference environment unavailable")
    result = (
        Benchmark(
            small_graph(),
            models=[model],
            seeds=[1],
            device=device,
            epochs=2,
            latency_warmup=1,
            latency_repeats=2,
            profiling={"level": "detailed", "epochs": [2]},
            output_dir=str(tmp_path),
        )
        .run()
        .records[0]
    )
    assert result["status"] == "SUCCESS", result.get("failure_reason")
    assert result["instrumented"]
    execution = result["worker_execution"] if model != "gcn" else result["execution"]
    # Parent preserves worker trace metadata as well as recovered crash-safe events.
    traces = execution.get("traces", [])
    assert len(traces) == 1, execution
    trace = json.loads(Path(traces[0]["path"]).read_text())
    assert any("egbench::training_forward" == e.get("name") for e in trace["traceEvents"])
    if device == "cuda":
        captured = any(e.get("cat") == "kernel" for e in trace["traceEvents"])
        assert captured or any("kernel tracing unavailable" in n for n in execution["notes"])
        assert any(e.get("cuda_elapsed_ms", 0) > 0 for e in execution["events"])
    if model == "graphgps":
        assert any(e["phase"] == "structural_preprocessing" for e in execution["events"])


def test_instrumentation_preserves_cpu_weights(tmp_path):
    states = []
    for level in ["none", "lightweight", "detailed"]:
        row = (
            Benchmark(
                small_graph(),
                models=["gcn"],
                seeds=[4],
                device="cpu",
                epochs=3,
                latency_warmup=1,
                latency_repeats=2,
                profiling={"level": level, "epochs": [2]},
                output_dir=str(tmp_path / level),
            )
            .run()
            .records[0]
        )
        assert row["status"] == "SUCCESS"
        states.append(torch.load(row["checkpoint_path"], weights_only=False)["model_state_dict"])
    for state in states[1:]:
        assert all(torch.equal(state[key], states[0][key]) for key in state)


@pytest.mark.parametrize("model", ["gcn", "graphgps"])
def test_progressive_selects_only_requested_stage(tmp_path, model):
    from efficientgraphbench import FidelityStage, ProgressiveBenchmark, ProgressiveConfig

    if model == "graphgps":
        from efficientgraphbench.models.catalog import model_spec

        if not Path(model_spec(model)["python"]).exists():
            pytest.skip("Pinned GraphGPS environment unavailable")
    result = ProgressiveBenchmark(
        small_graph(),
        models=[model],
        seeds=[1],
        device="cpu",
        output_dir=str(tmp_path),
        progressive=ProgressiveConfig(
            stages=[FidelityStage("initial", 0.5, 2), FidelityStage("full", 1, 2)]
        ),
        profiling={"level": "detailed", "epochs": [2], "stages": ["full"]},
    ).run()
    assert all(row["status"] == "SUCCESS" for row in result.records)
    key = "worker_execution" if model == "graphgps" else "execution"
    assert result.records[0][key].get("traces", []) == []
    assert len(result.records[1][key]["traces"]) == 1
    assert result.records[1]["parent_checkpoint_id"] == result.records[0]["checkpoint_id"]
    artifact = result.execution_analysis()
    assert artifact["report"].exists()


@pytest.mark.parametrize("exception", [torch.OutOfMemoryError, KeyboardInterrupt])
def test_partial_training_is_saved(tmp_path, monkeypatch, exception):
    from efficientgraphbench.benchmark import runner
    from efficientgraphbench.results.writer import read_results

    original = runner.build_model

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        count = 0

        def fail(module, inputs):
            nonlocal count
            count += 1
            if count == 3:
                raise exception("injected test failure")

        model.register_forward_pre_hook(fail)
        return model

    monkeypatch.setattr(runner, "build_model", build)
    bench = Benchmark(
        small_graph(),
        models=["gcn"],
        seeds=[1, 2],
        device="cpu",
        epochs=3,
        output_dir=str(tmp_path),
    )
    if exception is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            bench.run()
    else:
        bench.run()
    rows = read_results(tmp_path)
    first = rows[0]
    assert first["execution"]["last_completed_epoch"] == 1
    assert first["execution"]["failure_phase"] == "optimization_seconds"
    if exception is torch.OutOfMemoryError:
        assert first["status"] == "OOM"
        assert "execution" not in rows[1]
    else:
        assert first["interrupted"]
    assert first["execution"]["accounting"]["domains"][0]["measured_union_seconds"] > 0
