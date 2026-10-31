"""Portable, offline execution explorer. No model or training imports."""

import csv
import hashlib
import json
import shutil
import zipfile
from html import escape
from pathlib import Path

from efficientgraphbench.execution_analysis.comparison import device_comparisons
from efficientgraphbench.names import model_name
from efficientgraphbench.profiling.accounting import reconcile
from efficientgraphbench.profiling.trace_files import component
from efficientgraphbench.results.artifact import InteractiveReportOutput


def execution_report(records, output_dir, metadata=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sessions = []
    preview_budget = 2 * 1024 * 1024
    seen_sources = set()
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
                "stage_wall_seconds": row.get("stage_wall_time_sec"),
                "peak_gpu_reserved_mib": row.get("peak_gpu_reserved_mib"),
                "traces": [],
            }
            session["accounting"] = reconcile(source.get("events", []), row.get("status"))
            for trace in source.get("traces", []):
                path = Path(trace["path"])
                if not path.exists():
                    session.setdefault("notes", []).append("Trace file unavailable: " + path.name)
                    continue
                if path.resolve() in seen_sources:
                    raise ValueError(
                        "Duplicate source trace path; ambiguous attribution: " + str(path)
                    )
                seen_sources.add(path.resolve())
                folder = output / "traces"
                folder.mkdir(exist_ok=True)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if trace.get("sha256") and trace["sha256"] != digest:
                    raise ValueError("Source trace checksum mismatch: " + str(path))
                identity_key = json.dumps(
                    {
                        "source": str(path.resolve()),
                        "run": row.get("run_id"),
                        "model": row.get("model"),
                        "seed": row.get("seed"),
                        "stage": session["stage_name"],
                        "clock": source.get("clock_domain"),
                        "capture_id": trace.get("capture_id"),
                        "epoch": trace.get("epoch"),
                    },
                    sort_keys=True,
                )
                suffix = hashlib.sha256(identity_key.encode()).hexdigest()[:24]
                filename = (
                    "-".join(
                        [
                            "b" + component(trace.get("identity", {}).get("benchmark_id"))[:12],
                            component(row.get("model")),
                            "seed" + component(row.get("seed")),
                            "stage" + component(session["stage_name"]),
                            "epoch" + component(trace.get("epoch")),
                            component(origin),
                            component(trace.get("worker_attempt", source.get("clock_domain")))[:12],
                            suffix,
                        ]
                    )
                    + ".trace.json"
                )
                target = folder / filename
                copy_trace(path, target, digest)
                try:
                    data = json.loads(path.read_text(encoding="utf8"))
                except (ValueError, OSError) as exc:
                    session["notes"].append("Trace cannot be read: " + str(exc))
                    continue
                capture = data.get("efficientgraphbench")
                if capture:
                    if (
                        capture.get("capture_id") != trace.get("capture_id")
                        or capture.get("epoch") != trace.get("epoch")
                        or capture.get("identity") != trace.get("identity")
                        or capture.get("worker_attempt") != trace.get("worker_attempt")
                        or capture.get("identity", {}).get("seed") != row.get("seed")
                    ):
                        raise ValueError("Trace capture identity does not match its metadata/seed")
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
                        "sha256": digest,
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


def epoch_clock_rows(events):
    """Inclusive host wall intervals from the training process's monotonic journal."""
    children = {}
    for event in events:
        if event.get("phase") in {"optimization_seconds", "validation_seconds"}:
            children.setdefault(event.get("parent_event_id"), []).append(event)
    rows = []
    for event in events:
        if event.get("phase") != "epoch":
            continue
        row = {
            "epoch": event.get("epoch"),
            "status": event.get("status"),
            "start_utc": event.get("wall_start_utc"),
            "end_utc": event.get("wall_end_utc"),
            "epoch_wall_seconds": (
                event["duration_ns"] / 1e9 if event.get("duration_ns") is not None else None
            ),
        }
        for phase in ("optimization_seconds", "validation_seconds"):
            spans = [
                e
                for e in children.get(event["event_id"], [])
                if e["phase"] == phase and e.get("clock_domain") == event.get("clock_domain")
            ]
            row[phase] = (
                sum(e["duration_ns"] for e in spans) / 1e9
                if spans and all(e.get("duration_ns") is not None for e in spans)
                else None
            )
        rows.append(row)
    return sorted(rows, key=lambda row: row["epoch"] or 0)


def _write(payload, output):
    manifest = validate_trace_files(payload, output)
    payload["trace_manifest"] = manifest
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
        session["epoch_clock"] = epoch_clock_rows(session.get("events", []))
    clock_log = output / "training-clock.csv"
    with clock_log.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dataset",
                "model",
                "seed",
                "stage",
                "process",
                "clock_domain",
                "scope",
                "epoch",
                "status",
                "start_utc",
                "end_utc",
                "epoch_wall_seconds",
                "optimization_seconds",
                "validation_seconds",
                "training_seconds",
                "stage_wall_seconds",
            ],
        )
        writer.writeheader()
        for session in payload["sessions"]:
            if not session.get("model"):
                continue
            identity = {k: session.get(k) for k in ("dataset", "model", "seed", "clock_domain")}
            identity.update(stage=session.get("stage_name"), process=session.get("origin"))
            if session.get("origin") == "execution":
                writer.writerow(
                    {
                        **identity,
                        "scope": "stage" if session.get("stage_index") is not None else "run",
                        **{
                            k: session.get(k)
                            for k in ("status", "training_seconds", "stage_wall_seconds")
                        },
                    }
                )
            for row in session["epoch_clock"]:
                writer.writerow({**identity, "scope": "epoch", **row})
    (output / "execution.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf8")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    document = template.replace("__EXECUTION_DATA__", json.dumps(payload).replace("<", "\\u003c"))
    report = output / "execution.html"
    report.write_text(document, encoding="utf8")
    bundle = output / "execution-bundle.zip"
    files = [report, output / "execution.json", clock_log]
    for session in payload["sessions"]:
        for trace in session.get("traces", []):
            relative = Path(trace["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe trace path")
            target = output / relative
            if target.exists():
                files.append(target)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(output).as_posix())
        archive.writestr("trace-manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(bundle) as archive:
        names = [n for n in archive.namelist() if n.endswith(".trace.json")]
        if len(names) != len(set(names)) or set(names) != {t["path"] for t in manifest["files"]}:
            raise ValueError("ZIP trace contents do not match the declared manifest")
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
                copy_trace(source, target, trace.get("sha256"))
    return _write(payload, output)


def copy_trace(source, target, expected_hash=None):
    """Reuse identical files; never overwrite a different capture."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_hash and expected_hash != digest:
        raise ValueError("Trace checksum mismatch: " + str(source))
    try:
        with target.open("xb") as stream, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)
    except FileExistsError:
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Refusing to overwrite a different trace: " + str(target)) from None


def validate_trace_files(payload, output):
    paths, files, missing = set(), [], []
    for session in payload["sessions"]:
        for trace in session.get("traces", []):
            relative = Path(trace["path"])
            key = relative.as_posix().casefold()
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe trace path")
            if key in paths:
                raise ValueError("Duplicate trace path in report metadata: " + str(relative))
            paths.add(key)
            path = output / relative
            if not path.exists():
                if trace.get("available") is False:
                    missing.append(relative.as_posix())
                    continue
                raise ValueError("Declared trace file is missing: " + str(relative))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if trace.get("sha256") and digest != trace["sha256"]:
                raise ValueError("Packaged trace checksum mismatch: " + str(relative))
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "capture_id": trace.get("capture_id"),
                    "model": session.get("model"),
                    "seed": session.get("seed"),
                    "stage": session.get("stage_name"),
                    "epoch": trace.get("epoch"),
                }
            )
    return {
        "declared_references": len(paths),
        "packaged_files": len(files),
        "missing_files": missing,
        "files": files,
    }


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
