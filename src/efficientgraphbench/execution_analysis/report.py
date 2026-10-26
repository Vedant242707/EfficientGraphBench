"""Portable, offline execution explorer. No model or training imports."""

import json
import shutil
from html import escape
from pathlib import Path

from efficientgraphbench.execution_analysis.comparison import device_comparisons
from efficientgraphbench.names import model_name
from efficientgraphbench.profiling.accounting import reconcile
from efficientgraphbench.results.artifact import InteractiveReportOutput


def execution_report(records, output_dir, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sessions = []
    for row in records:
        if row.get("measured") is False or str(row.get("status", "")).startswith("SKIPPED"):
            continue
        for origin in ("execution", "worker_execution"):
            source = row.get(origin)
            if not source:
                continue
            session = {
                **source,
                "notes": list(source.get("notes", [])),
                "label": " · ".join(
                    [
                        model_name(row.get("model", "unknown")),
                        "seed " + str(row.get("seed", "—")),
                        "stage " + str(row.get("stage_index", "full")),
                        str(row.get("device", "—")),
                        str(row.get("status", "—")),
                        "reference worker"
                        if origin == "worker_execution"
                        else "controller attempt",
                    ]
                ),
                "num_nodes": row.get("num_nodes"),
                "num_edges": row.get("num_edges"),
                "hardware": row.get("physical_hardware", row.get("hardware")),
                "profiler_settings": row.get("profiler_settings"),
                "failure_reason": row.get("failure_reason"),
                "status": row.get("status"),
                "stage_index": row.get("stage_index"),
                "origin": origin,
                "device": row.get("device"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "training_seconds": row.get("training_time_sec"),
                "peak_gpu_reserved_mib": row.get("peak_gpu_reserved_mib"),
                "traces": [],
            }
            session["accounting"] = reconcile(source.get("events", []), row.get("status"))
            for trace in source.get("traces", []):
                path = Path(trace["path"])
                if not path.exists():
                    session.setdefault("notes", []).append("Trace file unavailable: " + path.name)
                    continue
                folder = output / "traces"
                folder.mkdir(exist_ok=True)
                target = folder / path.name
                if target.resolve() != path.resolve():
                    shutil.copyfile(path, target)
                data = json.loads(path.read_text(encoding="utf8"))
                measured = [
                    e
                    for e in data.get("traceEvents", [])
                    if e.get("ph") == "X" and isinstance(e.get("dur"), (float, int))
                ]
                session["traces"].append(
                    {
                        **trace,
                        "path": "traces/" + target.name,
                        "event_count": len(measured),
                        "preview": measured[:20000],
                    }
                )
            sessions.append(session)
    if metadata and metadata.get("execution"):
        sessions.insert(
            0, {**metadata["execution"], "label": "Complete benchmark lifecycle", "traces": []}
        )
    payload = {
        "schema_version": 1,
        "sessions": sessions,
        "device_comparisons": device_comparisons(records),
        "scope": "Saved events only. Clock domains cannot be aligned retrospectively.",
    }
    return _write(payload, output)


def _write(payload, output):
    (output / "execution.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    document = template.replace("__EXECUTION_DATA__", json.dumps(payload).replace("<", "\\u003c"))
    report = output / "execution.html"
    report.write_text(document, encoding="utf8")
    return InteractiveReportOutput(report=report, data=output / "execution.json")


def reopen_execution(path, output_dir=None):
    """Re-render a saved export without running a benchmark."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sessions"), list):
        raise ValueError("Unsupported execution export")
    output = Path(output_dir or path.parent)
    output.mkdir(parents=True, exist_ok=True)
    for session in payload["sessions"]:
        for trace in session.get("traces", []):
            relative = Path(trace["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Trace export must use a relative path inside its folder")
            source, target = path.parent / relative, output / relative
            if source.exists() and source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    return _write(payload, output)


def report_link(records, output, metadata=None):
    if not any(row.get("execution") or row.get("worker_execution") for row in records):
        return ""
    execution_report(records, Path(output) / "execution-analysis", metadata)
    return (
        "<section><h2>Execution Analysis</h2><p>Measured lifecycle, failure accounting "
        "and selected-epoch traces. GPU kernel information appears only when captured.</p>"
        '<p><a href="' + escape("execution-analysis/execution.html") + '">'
        "Open Execution Analysis</a></p></section>"
    )
