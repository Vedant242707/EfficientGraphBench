"""Record installed environments after successful reference smoke tests."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("graphgps", "sgformer", "pyg"):
    directory = ROOT / "environments" / name
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    python = ROOT / metadata["python"]
    packages = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.metadata as m; "
            "print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version "
            "for d in m.distributions())))",
        ],
        text=True,
    )
    (directory / "requirements-lock.txt").write_text(packages, encoding="utf-8")
    if name != "pyg":
        metadata["validated"] = True
        metadata["validation"] = "Cora CUDA reference worker smoke completed successfully"
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
