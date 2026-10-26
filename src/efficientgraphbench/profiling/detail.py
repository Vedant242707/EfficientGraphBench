"""Opt-in, selected-epoch profiling. No CUDA synchronization per operation."""

import contextvars
import json
from contextlib import contextmanager

from efficientgraphbench.profiling.events import CURRENT, span
from efficientgraphbench.profiling.sampling import sample_memory

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
            }
            recorder.profile_attempts.append(diagnostic)
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                candidate = torch.profiler.profile(
                    activities=activities, profile_memory=True, record_shapes=True
                )
                candidate.__enter__()
                profile = candidate
                diagnostic["status"] = "recording"
            except Exception as exc:
                diagnostic.update(status="start_failed", reason=str(exc))
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
                path = recorder.path.with_name(
                    recorder.path.stem + "-epoch" + str(number) + ".trace.json"
                )
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
                )
                if device.type == "cuda" and not kernels:
                    recorder.notes.append(
                        "CUDA kernel tracing unavailable: exported trace has no kernel events"
                    )
                recorder.artifacts.append(
                    {
                        "path": str(path),
                        "epoch": number,
                        "kernel_events": len(kernels),
                        "launch_events": len(launches),
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
