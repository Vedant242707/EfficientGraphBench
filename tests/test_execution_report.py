import csv
import json
import zipfile

from efficientgraphbench.execution_analysis import execution_report, reopen_execution
from efficientgraphbench.execution_analysis.comparison import device_comparisons


def test_worker_clock_preserves_failed_epoch_and_single_stage_total(tmp_path):
    def event(phase, event_id, parent, duration, epoch=1):
        return dict(
            phase=phase,
            event_id=event_id,
            parent_event_id=parent,
            clock_domain="worker",
            epoch=epoch,
            start_offset_ns=0,
            end_offset_ns=duration,
            duration_ns=duration,
            status="incomplete" if duration is None else "success",
        )

    events = [
        event("epoch", "a", None, 2000000000),
        event("optimization_seconds", "b", "a", 1000000000),
        event("epoch", "c", None, None, 2),
    ]
    report = execution_report(
        [
            dict(
                dataset="texas",
                model="graphgps",
                seed=2,
                stage_index=1,
                status="OOM",
                execution={"events": []},
                worker_execution={"events": events},
            )
        ],
        tmp_path,
    )
    rows = list(csv.DictReader((tmp_path / "training-clock.csv").read_text().splitlines()))
    assert [r["scope"] for r in rows] == ["stage", "epoch", "epoch"]
    assert rows[0]["training_seconds"] == ""
    assert rows[1]["epoch_wall_seconds"] == "2.0"
    assert rows[1]["optimization_seconds"] == "1.0"
    assert rows[1]["validation_seconds"] == ""
    assert rows[2]["epoch_wall_seconds"] == ""
    assert rows[2]["status"] == "incomplete"
    with zipfile.ZipFile(report["bundle"]) as bundle:
        assert bundle.read("training-clock.csv") == (tmp_path / "training-clock.csv").read_bytes()


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
