import json

import pytest

from efficientgraphbench.profiling.events import CURRENT, EventRecorder, read_journal, span


def test_nested_failure_and_recovery(tmp_path):
    recorder = EventRecorder(tmp_path / "events.jsonl", {"model": "gcn"})
    token = CURRENT.set(recorder)
    try:
        with pytest.raises(MemoryError), span("attempt"), span("training_forward"):
            raise MemoryError("out of memory")
    finally:
        recorder.close()
        CURRENT.reset(token)
    rows = read_journal(recorder.path)
    assert len(rows) == 2
    assert all(r["status"] == "oom" and r["duration_ns"] > 0 for r in rows)
    assert rows[1]["parent_event_id"] == rows[0]["event_id"]


def test_incomplete_is_not_zero(tmp_path):
    path = tmp_path / "crash.jsonl"
    path.write_text(
        json.dumps(
            {"schema_version": 1, "event_id": "x", "record_type": "start", "start_offset_ns": 17}
        )
        + "\n{"
    )
    (row,) = read_journal(path)
    assert row["status"] == "incomplete"
    assert row["duration_ns"] is None


def test_worker_crash_preserves_parent(tmp_path):
    import sys

    from efficientgraphbench.benchmark.execution import ExternalEnvironmentRunner, RunnerError

    script = tmp_path / "crash.py"
    script.write_text("import os\nos._exit(3)")
    with pytest.raises(RunnerError) as error:
        ExternalEnvironmentRunner([sys.executable, str(script)], tmp_path / "exchange").run({})
    measurements = error.value.progressive_measurements
    assert measurements["worker_wall_time_sec"] > 0
    assert measurements["worker_execution"]["events"] == []


def test_worker_crash_recovers_incremental_events(tmp_path):
    import sys
    from pathlib import Path

    from efficientgraphbench.benchmark.execution import ExternalEnvironmentRunner, RunnerError

    script = tmp_path / "partial.py"
    source = str(Path(__file__).resolve().parents[1] / "src")
    script.write_text(
        "import sys,json,os\n"
        f"sys.path.insert(0, {source!r})\n"
        "from efficientgraphbench.profiling.events import EventRecorder\n"
        'r=json.load(open(sys.argv[1]))["execution_context"]\n'
        'rec=EventRecorder(r["path"], r["identity"])\n'
        'with rec.span("worker_execution"):\n'
        '    with rec.span("completed_preprocessing"):\n'
        "        pass\n"
        "    os._exit(3)\n",
        encoding="utf8",
    )
    with pytest.raises(RunnerError) as error:
        ExternalEnvironmentRunner([sys.executable, str(script)], tmp_path / "exchange").run({})
    measured = error.value.progressive_measurements
    assert measured["worker_wall_time_sec"] > 0
    root, phase = measured["worker_execution"]["events"]
    assert root["duration_ns"] is None and root["status"] == "incomplete"
    assert phase["duration_ns"] >= 0 and phase["status"] == "success"


def test_exclusive_accounting():
    from efficientgraphbench.profiling.accounting import reconcile

    def event(identifier, phase, start, end, parent=None):
        return dict(
            event_id=identifier,
            parent_event_id=parent,
            clock_domain="one",
            phase=phase,
            status="success",
            start_offset_ns=start,
            end_offset_ns=end,
            duration_ns=end - start,
        )

    result = reconcile(
        [
            event("r", "attempt", 0, 100),
            event("a", "optimization_seconds", 10, 70, "r"),
            event("b", "training_forward", 20, 40, "a"),
        ]
    )
    (domain,) = result["domains"]
    assert domain["measured_union_seconds"] == 100 / 1e9
    assert domain["exclusive_seconds"]["optimization_seconds"] == 40 / 1e9
    assert domain["exclusive_seconds"]["training_forward"] == 20 / 1e9
    assert sum(domain["exclusive_seconds"].values()) == pytest.approx(100 / 1e9)


def test_different_process_clocks_are_not_added():
    from efficientgraphbench.profiling.accounting import reconcile

    events = [
        dict(
            event_id=str(i),
            parent_event_id=None,
            phase="attempt",
            clock_domain=str(i),
            status="success",
            start_offset_ns=0,
            end_offset_ns=100,
            duration_ns=100,
        )
        for i in range(2)
    ]
    assert len(reconcile(events)["domains"]) == 2
