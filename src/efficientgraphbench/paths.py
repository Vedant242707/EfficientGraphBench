"""Package resources are read-only; downloaded environments live outside the wheel."""

import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
RESOURCES = PACKAGE / "resources"
WORKER = PACKAGE / "runners/reference_worker.py"
CHECKOUT = PACKAGE.parents[1] if PACKAGE.parent.name == "src" else None


def home():
    configured = os.environ.get("EGBENCH_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    # Preserve already-installed developer environments, without requiring a checkout.
    if CHECKOUT and (CHECKOUT / "environments/graphgps/metadata.json").exists():
        return CHECKOUT
    if sys.platform == "win32":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
            / "EfficientGraphBench"
        )
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "efficientgraphbench"


def environment_dir(name):
    local = home() / "environments" / name
    return local if (local / "metadata.json").exists() else RESOURCES / "environments" / name
