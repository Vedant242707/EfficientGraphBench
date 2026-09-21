"""Framework-independent physical machine identity (also imported by workers)."""

import hashlib
import json
import platform
import subprocess
import sys

import psutil


def physical_hardware(device):
    info = {
        "machine": platform.node(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "os": platform.platform(),
        "device": str(device).split(":")[0],
        "gpu": None,
        "gpu_uuid": None,
        "total_vram_mib": None,
        "cuda_driver": None,
    }
    if info["device"] == "cuda":
        raw = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                text=True,
                timeout=15,
            )
            .strip()
            .split(",")
        )
        info.update(
            gpu=raw[0].strip(),
            gpu_uuid=raw[1].strip(),
            total_vram_mib=float(raw[2]),
            cuda_driver=raw[3].strip(),
        )
    stable = {k: v for k, v in info.items() if k != "cuda_driver"}
    # Python versions disagree on the Windows 10/11 marketing label for
    # the same kernel build. Keep the descriptive string in provenance,
    # but compare the native numeric version across isolated interpreters.
    if sys.platform == "win32":
        windows = sys.getwindowsversion()
        stable["os"] = f"Windows-{windows.major}.{windows.minor}.{windows.build}"
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return info, digest
