"""Install a pinned reference without changing the controller environment.

Usage: python scripts/setup_reference.py graphgps|sgformer
Requires Git and uv on PATH; installs inside this workspace.
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", choices=["graphgps", "sgformer"])
args = parser.parse_args()
folder = ROOT / "environments" / args.model
spec = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def run(*command):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


repository = ROOT / spec["repository"]
if not repository.exists():
    run("git", "clone", spec["repository_url"], repository)
    run("git", "-C", repository, "checkout", "--detach", spec["commit"])
else:
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != spec["commit"]:
        raise SystemExit(
            "Existing checkout is not the pinned commit; preserve it and resolve manually"
        )
python = ROOT / spec["python"]
if not python.exists():
    run(
        "uv",
        "python",
        "install",
        "3.10.20",
        "--install-dir",
        ROOT / ".python",
        "--cache-dir",
        ROOT / ".uv-cache",
    )
    run(
        "uv",
        "venv",
        folder / ".venv",
        "--python",
        ROOT / ".python/cpython-3.10.20-windows-x86_64-none/python.exe",
        "--cache-dir",
        ROOT / ".uv-cache",
    )
run(
    "uv",
    "pip",
    "install",
    "--python",
    python,
    "--cache-dir",
    ROOT / ".uv-cache",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu117",
    "--index-strategy",
    "unsafe-best-match",
    "-f",
    "https://data.pyg.org/whl/torch-1.13.0+cu117.html",
    "-r",
    folder / "requirements-lock.txt",
)
print("Installed pinned environment. Validate with egbench run before claiming availability.")
