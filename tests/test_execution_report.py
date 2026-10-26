import json

from efficientgraphbench.execution_analysis import execution_report, reopen_execution
from efficientgraphbench.execution_analysis.comparison import device_comparisons


def test_export_reopen_and_missing(tmp_path):
    report = execution_report([], tmp_path / "empty")
    assert "No saved execution events" in report["report"].read_text(encoding="utf8")
    reopened = reopen_execution(report["data"], tmp_path / "again")
    assert json.loads(reopened["data"].read_text())["sessions"] == []


def test_skip_does_not_duplicate_cost(tmp_path):
    report = execution_report(
        [{"status": "SKIPPED_AFTER_OOM", "execution": {"events": [], "notes": []}}], tmp_path
    )
    assert json.loads(report["data"].read_text())["sessions"] == []


def test_only_matching_cpu_gpu_scopes():
    base = dict(
        status="SUCCESS",
        model="gcn",
        seed=1,
        run_id="x",
        epochs_run=3,
        training_time_sec=2,
        execution_identity={"definition": {"recipe": "same"}},
        profiler_settings={"level": "lightweight"},
    )
    cpu, gpu = {**base, "device": "cpu"}, {**base, "device": "cuda", "training_time_sec": 1}
    assert device_comparisons([cpu, gpu])[0]["training_speedup"] == 2
    assert not device_comparisons([cpu, {**gpu, "profiler_settings": {"level": "detailed"}}])
    assert not device_comparisons([cpu, {**gpu, "status": "OOM"}])
