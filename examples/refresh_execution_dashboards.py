"""Update all saved execution HTML pages and their ZIP copies without running models."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from efficientgraphbench.execution_analysis.report import (
    bounded_trace_preview,
    dashboard_pages,
    epoch_clock_rows,
)


def refresh(source):
    payload = json.loads(source.read_text(encoding="utf8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sessions"), list):
        raise ValueError("Not an EfficientGraphBench execution export")
    for session in payload["sessions"]:
        session.setdefault("epoch_clock", epoch_clock_rows(session.get("events", [])))
    folder = source.parent
    bundle = folder / "execution-bundle.zip"
    traces = [t for s in payload["sessions"] for t in s.get("traces", [])]
    budget = (2 * 1024 * 1024) // max(1, len(traces))
    # Recover empty/truncated previews from preserved captures, not from training.
    archive = zipfile.ZipFile(bundle) if bundle.exists() else None
    try:
        for trace in traces:
            relative = Path(trace["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Trace must stay inside the export folder")
            target = folder / relative
            member = relative.as_posix()
            if target.is_file():
                raw = target.read_bytes()
            elif archive and member in archive.namelist():
                raw = archive.read(member)
            else:
                continue
            if trace.get("sha256") and hashlib.sha256(raw).hexdigest() != trace["sha256"]:
                raise ValueError("Trace checksum mismatch: " + member)
            trace["preview"] = bounded_trace_preview(json.loads(raw).get("traceEvents", []), budget)
    finally:
        if archive:
            archive.close()
    pages = dashboard_pages(payload)
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        tmp = Path(tmp)
        for name, html in pages.items():
            (tmp / name).write_text(html, encoding="utf8")
        if bundle.exists():
            # Copy every non-HTML member, including traces only available in the ZIP.
            # Never truncate the previous bundle while constructing its replacement.
            with (
                zipfile.ZipFile(bundle) as old,
                zipfile.ZipFile(tmp / bundle.name, "w", compression=zipfile.ZIP_DEFLATED) as new,
            ):
                for entry in old.infolist():
                    if entry.filename not in pages:
                        with old.open(entry) as src, new.open(entry, "w") as dst:
                            shutil.copyfileobj(src, dst)
                for name in pages:
                    new.write(tmp / name, name)
            os.replace(tmp / bundle.name, bundle)
        for name in pages:
            os.replace(tmp / name, folder / name)
    return folder / "execution.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, help="Experiment folders or execution.json")
    roots = parser.parse_args().roots
    project = Path(__file__).resolve().parents[1]
    roots = roots or [project / "experiments", project / "examples/experiments"]
    sources = sorted(
        {
            p.resolve()
            for root in roots
            for p in ([root] if root.is_file() else root.rglob("execution.json"))
            if p.name == "execution.json"
        }
    )
    if not sources:
        parser.exit(1, "No execution.json found. Pass your experiment folder explicitly.\n")
    failures = 0
    for src in sources:
        try:
            print(refresh(src), flush=True)
        except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
            failures += 1
            print(f"Could not update {src}: {exc}", flush=True)
    print(f"Updated {len(sources) - failures} of {len(sources)} execution dashboards.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
