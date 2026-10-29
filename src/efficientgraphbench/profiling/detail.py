"""Opt-in, selected-epoch profiling. No CUDA synchronization per operation."""

import contextvars
import json
import os
import platform
import sys
from contextlib import contextmanager

from efficientgraphbench.profiling.capabilities import gpu_capture_status, library_diagnostics
from efficientgraphbench.profiling.events import CURRENT, span
from efficientgraphbench.profiling.sampling import sample_memory
from efficientgraphbench.profiling.trace_files import capture_path

ACTIVE = contextvars.ContextVar("detailed_epoch", default=None)


def validate_options(options):
    if not isinstance(options, dict):
        raise ValueError("profiling must be a mapping")
    if set(options) - {"level", "epochs", "models", "seeds", "stages", "memory_sampling"}:
        raise ValueError("Unknown profiling option")
    if options.get("level", "lightweight") not in {"none", "lightweight", "detailed"}:
        raise ValueError(
            "Supported profiling levels: none, lightweight, detailed; Nsight unsupported"
        )
    for key in ("epochs", "seeds"):
        if key in options and (
            not isinstance(options[key], list)
            or any(type(x) is not int or x < (1 if key == "epochs" else 0) for x in options[key])
        ):
            raise ValueError(key + " must be a list of integers")
    for key in ("models", "stages"):
        if key in options and (
            not isinstance(options[key], list) or any(not isinstance(x, str) for x in options[key])
        ):
            raise ValueError(key + " must be a list of names")

    if "memory_sampling" in options:
        sampling = options["memory_sampling"]
        if not isinstance(sampling, dict) or set(sampling) - {"interval_ms", "max_samples"}:
            raise ValueError("memory_sampling accepts interval_ms and max_samples")
        for key, default, low, high in [
            ("interval_ms", 50, 10, 10000),
            ("max_samples", 1000, 2, 10000),
        ]:
            value = sampling.get(key, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"memory_sampling.{key} must be an integer in [{low}, {high}]")
        if options.get("level") != "detailed":
            raise ValueError("memory_sampling requires detailed profiling")


def selected(recorder, number):
    if recorder is None or recorder.options.get("level") != "detailed":
        return False
    opts = recorder.options
    return number in opts.get("epochs", [1]) and all(
        key not in opts or recorder.identity.get(field) in opts[key]
        for key, field in [("models", "model"), ("seeds", "seed"), ("stages", "stage")]
    )


@contextmanager
def epoch(number, device):
    recorder = CURRENT.get()
    if recorder is None:
        yield
        return
    import torch

    profile = None
    pending = []
    requested = selected(recorder, number)
    diagnostic = None
    token = ACTIVE.set((device, pending) if requested else None)
    try:
        if requested:
            diagnostic = {
                "epoch": number,
                "device": str(device),
                "status": "requested",
                "supported_activities": [str(a) for a in torch.profiler.supported_activities()],
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "kineto_available": torch.profiler.kineto_available(),
                "observed_devices": getattr(recorder, "observed_devices", None),
                "library_diagnostics": library_diagnostics(torch),
            }
            recorder.profile_attempts.append(diagnostic)
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                    diagnostic["gpu_name"] = torch.cuda.get_device_name(device)
                diagnostic["requested_activities"] = [str(a) for a in activities]
                candidate = torch.profiler.profile(
                    activities=activities, profile_memory=True, record_shapes=True
                )
                candidate.__enter__()
                profile = candidate
                diagnostic["status"] = "recording"
            except Exception as exc:
                diagnostic.update(status="start_failed", reason=str(exc))
                gpu_capture_status(diagnostic, [], device, os.environ.get("EGB_WORKER_LOG"))
                recorder.notes.append("Detailed profiler could not start: " + str(exc))
        with sample_memory(recorder if requested else None, device, f"epoch {number}"):
            with span(
                "epoch",
                epoch=number,
                device=str(device),
                timing_scope="wall interval; existing phase synchronization retained",
            ):
                yield
        recorder.last_completed_epoch = number
    finally:
        ACTIVE.reset(token)
        if requested:
            # One completion synchronization, never a synchronization per operation.
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    for event, start, end in pending:
                        event["cuda_elapsed_ms"] = start.elapsed_time(end)
                        event["cuda_scope"] = "current-stream CUDA event interval; not kernel sum"
                        recorder.emit({**event, "record_type": "end"})
            except Exception as exc:
                recorder.notes.append("CUDA event interval unavailable: " + str(exc))
        if profile is not None:
            try:
                profile.__exit__(None, None, None)
                path, capture_id = capture_path(recorder, number)
                # Reserve exclusively before letting the profiler write its file.
                # A collision must never replace any existing capture.
                path.open("x", encoding="utf8").close()
                profile.export_chrome_trace(str(path))
                trace = json.loads(path.read_text(encoding="utf8"))
                events = trace.get("traceEvents", [])
                kernels = [e for e in events if e.get("cat") == "kernel"]
                launches = [
                    e
                    for e in events
                    if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                    and "launch" in e.get("name", "").lower()
                ]
                diagnostic.update(
                    status="captured"
                    if kernels or device.type != "cuda"
                    else "cpu_only_no_cuda_kernels",
                    kernel_events=len(kernels),
                    cuda_runtime_events=sum(e.get("cat") == "cuda_runtime" for e in events),
                    cuda_driver_events=sum(e.get("cat") == "cuda_driver" for e in events),
                )
                trace["efficientgraphbench"] = {
                    "schema_version": 1,
                    "capture_id": capture_id,
                    "identity": recorder.identity,
                    "worker_attempt": recorder.domain,
                    "epoch": number,
                    "profiler": diagnostic,
                }
                if device.type == "cuda" and not kernels:
                    diagnostic["cuda_capture_reason"] = (
                        "Isolated profiler does not advertise CUDA activity support; "
                        "CUDA execution/event timing does not imply kernel trace support"
                        if str(torch.profiler.ProfilerActivity.CUDA)
                        not in diagnostic["supported_activities"]
                        else "CUDA activity was requested and advertised, but the export has no "
                        "kernel events. Inspect worker.log for CUPTI/runtime errors; "
                        "this does not establish CPU execution"
                    )
                    recorder.notes.append(
                        "CUDA kernel tracing unavailable: " + diagnostic["cuda_capture_reason"]
                    )
                gpu_capture_status(diagnostic, kernels, device, os.environ.get("EGB_WORKER_LOG"))
                diagnostic["library_diagnostics"] = library_diagnostics(torch)
                path.write_text(json.dumps(trace), encoding="utf8")
                recorder.artifacts.append(
                    {
                        "path": str(path),
                        "capture_id": capture_id,
                        "identity": dict(recorder.identity),
                        "worker_attempt": recorder.domain,
                        "epoch": number,
                        "kernel_events": len(kernels),
                        "launch_events": len(launches),
                        "cuda_runtime_events": diagnostic["cuda_runtime_events"],
                        "cuda_driver_events": diagnostic["cuda_driver_events"],
                        "kernel_names": list(dict.fromkeys(e["name"] for e in kernels))[:20],
                        "streams": sorted({str(e.get("args", {}).get("stream")) for e in kernels}),
                        "format": "pytorch_chrome_trace",
                        "device": str(device),
                        "source": "torch.profiler exported Chrome trace",
                    }
                )
            except Exception as exc:
                diagnostic.update(status="export_failed", reason=str(exc))
                recorder.notes.append("Profiler export unavailable: " + str(exc))


@contextmanager
def operation(name):
    active = ACTIVE.get()
    if active is None:
        yield
        return
    import torch

    device, pending = active
    with span(
        name, timing_scope="Host call wall; asynchronous dispatch on CUDA; CUDA duration separate"
    ) as event:
        with torch.profiler.record_function("egbench::" + name):
            start = end = None
            if device.type == "cuda":
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record(torch.cuda.current_stream(device))
            try:
                yield
            finally:
                if end is not None:
                    end.record(torch.cuda.current_stream(device))
                    pending.append((event, start, end))
                    event["gpu_allocated_bytes"] = torch.cuda.memory_allocated(device)
                    event["gpu_reserved_bytes"] = torch.cuda.memory_reserved(device)
