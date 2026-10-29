"""Read-only profiler diagnostics; never load or replace a CUDA/CUPTI library."""

import os
import re
import sys
from pathlib import Path


def library_diagnostics(torch):
    roots = [Path(torch.__file__).parent / "lib"]
    for entry in sys.path:
        root = Path(entry) / "nvidia" / "cuda_cupti" / "lib"
        if root.is_dir():
            roots.append(root)
    for key in ("LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME"):
        for entry in os.environ.get(key, "").split(os.pathsep):
            if entry:
                roots.extend([Path(entry), Path(entry) / "extras/CUPTI/lib64"])
    candidates = sorted({str(p) for root in roots if root.is_dir() for p in root.glob("*cupti*")})
    loaded = []
    try:
        import psutil

        loaded = sorted(
            {
                m.path
                for m in psutil.Process().memory_maps()
                if any(x in m.path.lower() for x in ("cupti", "cudart", "libcuda", "kineto"))
            }
        )
        error = None
    except Exception as exc:
        error = str(exc)
    return {
        "library_search_environment": {
            key: os.environ.get(key)
            for key in ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_PATH", "CUDA_HOME")
        },
        "cupti_candidates": candidates,
        "candidate_scope": "torch/lib, pip NVIDIA CUPTI package and CUDA/library environment paths",
        "loaded_profiler_cuda_libraries": loaded,
        "loaded_library_inspection_error": error,
        "kineto_version": None,
        "kineto_version_note": "Bundled with PyTorch; no separate version exposed by public API",
    }


def cupti_log_errors(path):
    """Bounded read of native stderr redirected by the worker launcher."""
    if not path:
        return []
    try:
        with Path(path).open("rb") as stream:
            head = stream.read(131072)
            stream.seek(max(0, stream.seek(0, 2) - 131072))
            tail = stream.read(131072)
        lines = (head + b"\n" + tail).decode("utf8", errors="replace").splitlines()
        # Exclude generic 'If you see CUPTI_ERROR_...' permission advice.
        return list(
            dict.fromkeys(
                line[:2000]
                for line in lines
                if "if you see" not in line.lower()
                and (re.search(r"CUPTI_ERROR_\w+", line) or "CUPTI initialization failed" in line)
            )
        )[:20]
    except OSError:
        return []


def gpu_capture_status(diagnostic, kernels, device, log_path=None):
    errors = cupti_log_errors(log_path)
    diagnostic["cupti_errors"] = errors
    diagnostic["cupti_error_scope"] = "worker process log; may precede the selected epoch"
    diagnostic["gpu_profiling_status"] = (
        "available"
        if kernels
        else "unavailable"
        if str(device).startswith("cuda")
        else "not_requested"
    )
    if not kernels and errors:
        diagnostic["cuda_capture_reason"] = "GPU activity capture unavailable: " + " | ".join(
            errors
        )
