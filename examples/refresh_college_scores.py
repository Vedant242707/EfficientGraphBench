"""Refresh score pages from saved college notebook results without running models."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from efficientgraphbench.progressive.results import ProgressiveResults
from efficientgraphbench.results.dashboard import load_analysis_results


def refresh(batch):
    found = False
    for saved in sorted(batch.glob("*/full/raw/results.jsonl")):
        found = True
        result = load_analysis_results(saved.parent.parent).show_all(display=False)
        print(result["report"], result["dashboard"], sep="\n")
    for saved in sorted(batch.glob("*/progressive/**/progressive-results.json")):
        found = True
        data = json.loads(saved.read_text(encoding="utf8"))
        # show_all normally saves run records. Rebuild only the two HTML artifacts in place.
        with tempfile.TemporaryDirectory() as tmp:
            path = ProgressiveResults(data["records"], Path(tmp), data["metadata"]).show_all(
                display=False
            )
            for artifact in [path, path.with_name("benchmark-dashboard.html")]:
                dest = saved.parent / artifact.name
                shutil.copyfile(artifact, dest)
                print(dest)
    if not found:
        raise SystemExit(f"No saved college Full or Progressive results found in {batch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", nargs="?", type=Path, help="A college-reports-* folder")
    batch = parser.parse_args().batch
    if batch is None:
        root = Path(__file__).resolve().parents[1]
        choices = [
            p
            for folder in (root / "experiments", root / "examples/experiments")
            for p in folder.glob("college-reports-*")
            if p.is_dir()
        ]
        if not choices:
            raise SystemExit("Pass the path to a saved college-reports-* folder.")
        batch = max(choices, key=lambda p: p.name)
    print("Refreshing score reports in:", batch)
    refresh(batch)
