"""Pinned, isolated reference environments; no activation required."""

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from efficientgraphbench.paths import RESOURCES, environment_dir, home

NAMES = ("graphgps", "sgformer", "graphormer")


def environment_spec(name):
    name = name.lower().replace("sgformer_reference", "sgformer")
    if name not in NAMES:
        raise ValueError(f"Unknown reference environment: {name}")
    template = json.loads((RESOURCES / "environments" / name / "metadata.json").read_text())
    folder = environment_dir(name)
    installed = folder != RESOURCES / "environments" / name
    spec = json.loads((folder / "metadata.json").read_text())
    spec["commit"] = template.get("commit")
    if name != "graphormer":
        spec["repository"] = str(home() / template["repository"])
        spec["python"] = str(
            home()
            / "environments"
            / name
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        spec["validated"] = installed and spec.get("validated", False)
    return spec


def inspect_environment(name, verify=False):
    spec = environment_spec(name)
    result = dict(spec)
    if name == "graphormer":
        result["status"] = "UNSUPPORTED_TASK"
        return result
    if not Path(spec["python"]).exists() or not Path(spec["repository"]).exists():
        result.update(status="ENVIRONMENT_NOT_INSTALLED", validated=False)
        return result
    if verify:
        repo = Path(spec["repository"])
        base = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
        commit = subprocess.check_output([*base, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            [*base, "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if commit != spec["commit"] or dirty:
            raise ValueError("Reference checkout differs from pinned clean commit")
        source = (
            f"import torch, torch_geometric, sys, json; sys.path.insert(0, {str(repo)!r}); "
            if name == "graphgps"
            else "import torch, torch_geometric, sys, json; "
            f"sys.path.insert(0, {str(repo / 'medium')!r}); "
        )
        source += (
            "from graphgps.network.gps_model import GPSModel; "
            if name == "graphgps"
            else "from ours import SGFormer; "
        )
        source += "print(json.dumps(dict(python=sys.version, torch=torch.__version__, pyg=torch_geometric.__version__, cuda=torch.version.cuda)))"  # noqa: E501
        checked = subprocess.run(
            [spec["python"], "-c", source], capture_output=True, text=True, timeout=120
        )
        if checked.returncode:
            raise RuntimeError(checked.stderr[-4000:])
        result["verified_software"] = json.loads(checked.stdout.strip().splitlines()[-1])
        result["validated"] = True
    result["status"] = "READY" if result.get("validated") else "UNVERIFIED"
    return result


def setup_environment(name):
    name = name.lower()
    if name not in NAMES:
        raise ValueError(f"Choose one of: {', '.join(NAMES)}")
    spec = environment_spec(name)
    if name == "graphormer":
        return inspect_environment(name)
    if Path(spec["python"]).exists() and Path(spec["repository"]).exists():
        result = inspect_environment(name, verify=True)
    else:
        if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise ValueError(
                "The bundled research locks currently support Windows x86-64 only. Core PyG models remain available."  # noqa: E501
            )
        uv, git = shutil.which("uv"), shutil.which("git")
        if not uv or not git:
            raise RuntimeError(
                "Reference setup requires Git and uv on PATH. Install uv with: python -m pip install uv"  # noqa: E501
            )
        folder = home() / "environments" / name
        folder.mkdir(parents=True, exist_ok=True)
        repo = Path(spec["repository"])
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([git, "clone", spec["repository_url"], str(repo)], check=True)
            subprocess.run(
                [git, "-C", str(repo), "checkout", "--detach", spec["commit"]], check=True
            )
        else:
            actual = subprocess.check_output(
                [git, "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != spec["commit"]:
                raise ValueError(
                    "Existing repository is not at the pinned commit; it has been preserved"
                )
        subprocess.run([uv, "venv", str(folder / ".venv"), "--python", "3.10.20"], check=True)
        lock = RESOURCES / "environments" / name / "requirements-lock.txt"
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                spec["python"],
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu117",
                "--index-strategy",
                "unsafe-best-match",
                "-f",
                "https://data.pyg.org/whl/torch-1.13.0+cu117.html",
                "-r",
                str(lock),
            ],
            check=True,
        )
        for filename in ("requirements-lock.txt", "environment.yml"):
            shutil.copyfile(RESOURCES / "environments" / name / filename, folder / filename)
        spec["validated"] = False
        (folder / "metadata.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        result = inspect_environment(name, verify=True)
    result["validation"] = (
        "Pinned clean checkout and model import verified; benchmark success depends on task and hardware"  # noqa: E501
    )
    (home() / "environments" / name / "metadata.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
