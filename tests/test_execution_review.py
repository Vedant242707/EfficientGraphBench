import hashlib
import json
import time
import zipfile
from pathlib import Path

import pytest

from efficientgraphbench.execution_analysis import execution_report, reopen_execution
from efficientgraphbench.execution_analysis.report import profiling_status
from efficientgraphbench.profiling.detail import validate_options
from efficientgraphbench.profiling.events import EventRecorder, read_memory_samples
from efficientgraphbench.profiling.sampling import sample_memory


def test_bounded_samples_survive_failure(tmp_path):
    recorder = EventRecorder(
        tmp_path / "events.jsonl",
        {},
        {
            "level": "detailed",
            "memory_sampling": {"interval_ms": 10, "max_samples": 4},
        },
    )
    with pytest.raises(MemoryError), sample_memory(recorder, "cpu", "simulated worker"):
        time.sleep(0.08)
        raise MemoryError("simulated OOM; no hardware exhaustion")
    recorder.close()
    samples = read_memory_samples(recorder.path)
    assert len(samples) == 4
    assert samples[-1]["boundary"] == "failure"
    assert all(s["cpu_rss_bytes"] > 0 for s in samples)
    assert samples[-1]["offset_ns"] >= samples[0]["offset_ns"]


@pytest.mark.parametrize(
    "sampling",
    [
        {"interval_ms": 0},
        {"max_samples": 10001},
        {"max_samples": True},
        {"unknown": 1},
    ],
)
def test_sampling_limits(sampling):
    with pytest.raises(ValueError):
        validate_options({"level": "detailed", "memory_sampling": sampling})


def test_profiling_disabled_vs_failed():
    assert "disabled" in profiling_status({"instrumentation": {}})
    assert "start_failed" in profiling_status(
        {
            "instrumentation": {"level": "detailed"},
            "profile_attempts": [{"status": "start_failed", "reason": "test failure"}],
        }
    )


def test_profiler_start_failure_does_not_skip_work(tmp_path, monkeypatch):
    import torch

    from efficientgraphbench.profiling.detail import epoch, operation
    from efficientgraphbench.profiling.events import CURRENT

    def unavailable(**kwargs):
        raise RuntimeError("simulated profiler unavailable")

    monkeypatch.setattr(torch.profiler, "profile", unavailable)
    recorder = EventRecorder(tmp_path / "profile.jsonl", {}, {"level": "detailed"})
    token = CURRENT.set(recorder)
    try:
        with epoch(1, torch.device("cpu")), operation("training_forward"):
            value = torch.ones(3).sum().item()
        assert value == 3
        assert recorder.last_completed_epoch == 1
        assert recorder.profile_attempts[0]["status"] == "start_failed"
        assert any(e["phase"] == "training_forward" for e in recorder.events)
        assert not recorder.artifacts
    finally:
        CURRENT.reset(token)
        recorder.close()


def test_bundle_moved_preserves_full_trace_and_bounded_preview(tmp_path):
    trace = tmp_path / "original.trace.json"
    # Synthetic fixture verifies serialization only, not CUDA capture.
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "cat": "cpu_op", "name": "fixture", "ts": i, "dur": 1}
                    for i in range(3000)
                ]
            }
        )
    )
    result = execution_report(
        [
            {
                "model": "gcn",
                "status": "SUCCESS",
                "seed": 1,
                "execution": {
                    "events": [],
                    "instrumentation": {"level": "detailed"},
                    "traces": [{"path": str(trace), "epoch": 1}],
                },
            }
        ],
        tmp_path / "export",
    )
    target = tmp_path / "other-machine"
    with zipfile.ZipFile(result["bundle"]) as archive:
        archive.extractall(target)
    source_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    trace.unlink()
    reopened = reopen_execution(target / "execution.json", target / "reopened")
    session = json.loads(reopened["data"].read_text())["sessions"][0]
    exported = session["traces"][0]
    assert len(exported["preview"]) == 2000
    assert exported["event_count"] == 3000
    moved = reopened["report"].parent / exported["path"]
    assert hashlib.sha256(moved.read_bytes()).hexdigest() == source_hash == exported["sha256"]


def test_worker_oom_preserves_phases_samples_and_wall(tmp_path):
    import sys

    from efficientgraphbench.benchmark.execution import ExternalEnvironmentRunner

    script = tmp_path / "worker.py"
    source = str(Path(__file__).resolve().parents[1] / "src")
    script.write_text(
        f"import sys,json,time\nsys.path.insert(0,{source!r})\n"
        "from efficientgraphbench.profiling.events import EventRecorder\n"
        "from efficientgraphbench.profiling.sampling import sample_memory\n"
        "r=json.load(open(sys.argv[1]));c=r['execution_context']\n"
        "rec=EventRecorder(c['path'],c['identity'],c['options'])\n"
        "try:\n"
        " with rec.span('worker_execution'), sample_memory(rec,'cpu','worker'):\n"
        "  with rec.span('completed_preprocessing'): time.sleep(.02)\n"
        "  raise MemoryError('controlled simulated OOM')\n"
        "except MemoryError:\n"
        " result={k:r.get(k) for k in "
        "['run_id','model_id','seed','split_hash','canonical_graph_hash']}\n"
        " result.update(exchange_schema_version=1,status='OOM',"
        "worker_execution=rec.summary(),failure_reason='controlled simulated OOM')\n"
        " json.dump(result,open(sys.argv[2],'w'))\n"
        "finally: rec.close()\n"
    )
    row = ExternalEnvironmentRunner([sys.executable, str(script)], tmp_path / "exchange").run(
        {
            "config": {
                "profiling": {
                    "level": "detailed",
                    "memory_sampling": {
                        "interval_ms": 10,
                        "max_samples": 10,
                    },
                }
            },
        }
    )
    assert row["status"] == "OOM" and row["worker_wall_time_sec"] > 0
    assert any(
        e["phase"] == "completed_preprocessing" and e["duration_ns"] > 0
        for e in row["worker_execution"]["events"]
    )
    assert row["worker_execution"]["memory_samples"][-1]["boundary"] == "failure"
    root = next(e for e in row["worker_execution"]["events"] if e["phase"] == "worker_execution")
    assert root["status"] == "oom" and root["duration_ns"] > 0
