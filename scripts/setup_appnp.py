"""Fetch PyG's pinned APPNP reference class; keeps the installed PyG environment."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "79d33965a40b7fa83616a9f598a0f8619f25d939"
repo = ROOT / "external/PyG"
if not repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/pyg-team/pytorch_geometric",
            str(repo),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", COMMIT], check=True)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if actual != COMMIT:
    raise SystemExit(
        "Existing PyG checkout differs from the pinned commit; preserve and resolve manually"
    )
subprocess.run(["git", "-C", str(repo), "sparse-checkout", "add", "benchmark/citation"], check=True)
print("APPNP reference source installed; model loading verifies its exact normalized-text SHA256.")
