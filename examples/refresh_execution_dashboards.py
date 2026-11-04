"""Update all saved execution HTML pages and their ZIP copies without running models."""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from efficientgraphbench.execution_analysis.report import dashboard_pages, epoch_clock_rows


def refresh(source):
    payload = json.loads(source.read_text(encoding="utf8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sessions"), list):
        raise ValueError("Not an EfficientGraphBench execution export")
    for session in payload["sessions"]:
        session.setdefault("epoch_clock", epoch_clock_rows(session.get("events", [])))
    pages = dashboard_pages(payload)
    folder = source.parent
    bundle = folder / "execution-bundle.zip"
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
