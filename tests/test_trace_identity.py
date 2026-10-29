import hashlib
import json
import zipfile

import pytest

from efficientgraphbench.execution_analysis import execution_report, reopen_execution
from efficientgraphbench.execution_analysis.report import copy_trace, validate_trace_files
from efficientgraphbench.profiling.events import EventRecorder
from efficientgraphbench.profiling.trace_files import capture_path


def record(seed, path):
    return {
        "model": "graphgps",
        "seed": seed,
        "status": "SUCCESS",
        "worker_execution": {
            "events": [],
            "instrumentation": {"level": "detailed"},
            "traces": [{"path": str(path), "epoch": 10}],
        },
    }


def test_same_worker_basename_preserves_independent_seeds(tmp_path):
    rows = []
    for seed in [1, 2]:
        directory = tmp_path / str(seed)
        directory.mkdir()
        path = directory / "execution-worker-epoch10.trace.json"
        path.write_text(json.dumps({"traceEvents": [], "fixture_seed": seed}))
        rows.append(record(seed, path))
    result = execution_report(rows, tmp_path / "report")
    data = json.loads(result["data"].read_text())
    refs = [s["traces"][0] for s in data["sessions"]]
    assert len({t["path"] for t in refs}) == 2
    with zipfile.ZipFile(result["bundle"]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".trace.json")]
        assert len(names) == len(set(names)) == data["trace_manifest"]["declared_references"] == 2
        for seed, trace in enumerate(refs, 1):
            raw = archive.read(trace["path"])
            assert json.loads(raw)["fixture_seed"] == seed
            assert hashlib.sha256(raw).hexdigest() == trace["sha256"]


def test_repeated_source_trace_rejected(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text('{"traceEvents": []}')
    with pytest.raises(ValueError, match="Duplicate source trace"):
        execution_report([record(1, path), record(2, path)], tmp_path / "report")


def test_capture_identity_cannot_be_attributed_to_another_seed(tmp_path):
    path = tmp_path / "capture.trace.json"
    capture = {"capture_id": "id", "epoch": 10, "identity": {"seed": 1}, "worker_attempt": "worker"}
    path.write_text(json.dumps({"traceEvents": [], "efficientgraphbench": capture}))
    row = record(2, path)
    row["worker_execution"]["traces"][0].update(capture)
    with pytest.raises(ValueError, match="capture identity"):
        execution_report([row], tmp_path / "export")


def test_duplicate_declared_trace_rejected(tmp_path):
    trace = tmp_path / "x.trace.json"
    trace.write_text("{}")
    data = {"sessions": [{"traces": [{"path": trace.name}, {"path": trace.name}]}]}
    with pytest.raises(ValueError, match="Duplicate trace path"):
        validate_trace_files(data, tmp_path)


def test_declared_missing_and_tampered_traces_rejected(tmp_path):
    data = {"sessions": [{"traces": [{"path": "x.trace.json", "sha256": "wrong"}]}]}
    with pytest.raises(ValueError, match="missing"):
        validate_trace_files(data, tmp_path)
    (tmp_path / "x.trace.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_trace_files(data, tmp_path)


def test_trace_copy_never_overwrites_other_capture(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("new capture")
    target.write_text("existing capture")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        copy_trace(source, target)
    assert target.read_text() == "existing capture"


def test_capture_names_unique_for_attempt_and_epoch(tmp_path):
    paths = []
    for seed in [0, 1, 2]:
        r = EventRecorder(
            tmp_path / f"{seed}.jsonl",
            {
                "benchmark_id": "bench",
                "model": "graphgps",
                "seed": seed,
                "stage": "full",
                "role": "worker",
            },
        )
        try:
            for epoch in [10, 100, 10]:
                path, capture_id = capture_path(r, epoch)
                assert f"seed{seed}" in path.name and f"epoch{epoch}" in path.name
                assert capture_id in path.name
                paths.append(path)
        finally:
            r.close()
    assert len(set(paths)) == 9


def test_reopen_rejects_overwritten_legacy_trace(tmp_path):
    path = tmp_path / "x.trace.json"
    path.write_text("{}")
    data = {
        "schema_version": 1,
        "sessions": [
            {
                "traces": [
                    {
                        "path": path.name,
                        "sha256": "the-earlier-seeds-hash",
                    }
                ]
            }
        ],
    }
    meta = tmp_path / "execution.json"
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        reopen_execution(meta)
