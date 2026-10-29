"""Compare profiler environments in fresh processes without changing their packages."""

import argparse
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def probe(output):
    import torch

    from efficientgraphbench.profiling.capabilities import library_diagnostics

    result = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "kineto_available": torch.profiler.kineto_available(),
        "supported_activities": [str(a) for a in torch.profiler.supported_activities()],
        "build_config": torch.__config__.show(),
        "before": library_diagnostics(torch),
    }
    try:
        result["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["nvidia_smi_error"] = str(exc)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU substitution")
        x = torch.ones((256, 256), device="cuda")
        torch.cuda.synchronize()
        result["tensor_device"] = str(x.device)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        result["requested_activities"] = [str(a) for a in activities]
        with torch.profiler.profile(activities=activities) as profiler:
            for _ in range(3):
                x = torch.mm(x, x)
            torch.cuda.synchronize()
        trace = output.with_name(output.stem + "-" + uuid.uuid4().hex + ".trace.json")
        trace.open("x").close()
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text(encoding="utf8"))["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        result.update(
            trace=str(trace),
            kernel_events=len(kernels),
            kernel_names=list(dict.fromkeys(e["name"] for e in kernels))[:10],
            cuda_runtime_events=sum(e.get("cat") == "cuda_runtime" for e in events),
            gpu_profiling_status="available" if kernels else "unavailable",
        )
    except Exception as exc:
        result.update(gpu_profiling_status="unavailable", error=str(exc))
    result["after"] = library_diagnostics(torch)
    output.write_text(json.dumps(result, indent=2), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/graphgps-profiler-diagnostics")
    parser.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        probe(args.probe)
        return
    from efficientgraphbench.models.catalog import model_spec
    from efficientgraphbench.profiling.capabilities import cupti_log_errors

    folder = Path(args.output).resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True)
    summary = {}
    for label, executable in [
        ("main", sys.executable),
        ("graphgps", model_spec("graphgps")["python"]),
    ]:
        env = os.environ.copy()
        # Match the existing worker launcher's allocator compatibility policy.
        if label == "graphgps":
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            env.pop("PYTORCH_ALLOC_CONF", None)
        path = folder / (label + ".json")
        log = folder / (label + ".log")
        with log.open("w", encoding="utf8") as stream:
            try:
                process = subprocess.run(
                    [str(executable), str(Path(__file__).resolve()), "--probe", str(path)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=120,
                )
                result = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
                result["returncode"] = process.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                result = {"error": str(exc)}
        result["cupti_errors"] = cupti_log_errors(log)
        path.write_text(json.dumps(result, indent=2), encoding="utf8")
        summary[label] = result
        print(
            label,
            result.get("torch_version"),
            result.get("cuda_build"),
            result.get("gpu_profiling_status", "probe failed"),
            "kernels:",
            result.get("kernel_events", 0),
        )
    (folder / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("Diagnostics and native stderr:", folder)


if __name__ == "__main__":
    main()
