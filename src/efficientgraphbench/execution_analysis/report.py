"""Portable, offline execution explorer. No model or training imports."""

import hashlib
import json
import shutil
import zipfile
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
    preview_budget = 2 * 1024 * 1024
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
                "dataset": row.get("dataset"),
                "num_nodes": row.get("num_nodes"),
                "num_edges": row.get("num_edges"),
                "hardware": row.get("physical_hardware", row.get("hardware")),
                "profiler_settings": row.get("profiler_settings"),
                "failure_reason": row.get("failure_reason"),
                "status": row.get("status"),
                "stage_index": row.get("stage_index"),
                "stage_name": next(
                    (e.get("stage") for e in source.get("events", []) if e.get("stage")), "full"
                ),
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
                try:
                    data = json.loads(path.read_text(encoding="utf8"))
                except (ValueError, OSError) as exc:
                    session["notes"].append("Trace cannot be read: " + str(exc))
                    continue
                measured = [
                    e
                    for e in data.get("traceEvents", [])
                    if e.get("ph") == "X" and isinstance(e.get("dur"), (float, int))
                ]
                # A global byte budget prevents many traces from bloating HTML.
                preferred = [
                    e
                    for e in measured
                    if e.get("cat") in {"kernel", "cuda_runtime", "cuda_driver", "user_annotation"}
                ]
                rest = [
                    e
                    for e in measured
                    if e.get("cat")
                    not in {"kernel", "cuda_runtime", "cuda_driver", "user_annotation"}
                ]
                preview = []
                for event in preferred + rest:
                    size = len(json.dumps(event).encode("utf8"))
                    if size > preview_budget or len(preview) >= 2000:
                        continue
                    preview.append(event)
                    preview_budget -= size
                preview.sort(key=lambda e: e.get("ts", 0))
                session["traces"].append(
                    {
                        **trace,
                        "path": "traces/" + target.name,
                        "event_count": len(measured),
                        "preview": preview,
                        "size_bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
            session["profiling_status"] = profiling_status(session)
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


def profiling_status(session):
    options = session.get("instrumentation", {})
    if options.get("level", "lightweight") != "detailed":
        return "Detailed profiling disabled (" + options.get("level", "lightweight") + ")"
    if session.get("traces"):
        if any(t.get("available") is False for t in session["traces"]):
            return "Trace preview retained, but some full trace files are missing from this export"
        return "Detailed traces exported; CUDA kernels verified only when kernel count is positive"
    attempts = session.get("profile_attempts", [])
    if any(a.get("status") in {"captured", "cpu_only_no_cuda_kernels"} for a in attempts):
        return "Capture completed, but the trace file is unavailable; inspect collection notes"
    if attempts:
        return "; ".join(str(a.get("status")) + ": " + str(a.get("reason", "")) for a in attempts)
    if session.get("notes"):
        return "No accessible detailed trace; inspect diagnostic notes"
    return "Detailed requested, but no capture recorded; check epoch/model/seed/stage selection"


def _write(payload, output):
    # Upgrade display metadata from recorded events, including older exports.
    for session in payload["sessions"]:
        for key, event_key in [
            ("dataset", "dataset"),
            ("device", "device"),
            ("stage_name", "stage"),
        ]:
            if session.get(key) is None:
                session[key] = next(
                    (
                        e.get(event_key)
                        for e in session.get("events", [])
                        if e.get(event_key) is not None
                    ),
                    None,
                )
        if session.get("model"):
            session["label"] = " · ".join(
                [
                    model_name(session["model"]),
                    "seed " + str(session.get("seed")),
                    "stage "
                    + str(session.get("stage_name") or session.get("stage_index") or "full"),
                    str(session.get("device") or "Unavailable"),
                    str(session.get("status")),
                    "reference worker"
                    if session.get("origin") == "worker_execution"
                    else "controller attempt",
                ]
            )
        session["profiling_status"] = profiling_status(session)
    (output / "execution.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    document = template.replace("__EXECUTION_DATA__", json.dumps(payload).replace("<", "\\u003c"))
    report = output / "execution.html"
    report.write_text(document, encoding="utf8")
    bundle = output / "execution-bundle.zip"
    files = [report, output / "execution.json"]
    for session in payload["sessions"]:
        for trace in session.get("traces", []):
            relative = Path(trace["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe trace path")
            target = output / relative
            if target.exists():
                files.append(target)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in dict.fromkeys(files):
            archive.write(path, path.relative_to(output).as_posix())
    return InteractiveReportOutput(report=report, data=output / "execution.json", bundle=bundle)


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
            trace["available"] = source.exists()
            if not trace["available"]:
                session.setdefault("notes", []).append("Full trace missing: " + str(relative))
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
