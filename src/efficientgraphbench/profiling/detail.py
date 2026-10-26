"""Opt-in, selected-epoch profiling. No CUDA synchronization per operation."""

import contextvars
import json
from contextlib import contextmanager

from efficientgraphbench.profiling.events import CURRENT, span

ACTIVE = contextvars.ContextVar("detailed_epoch", default=None)


def validate_options(options):
    if not isinstance(options, dict):
        raise ValueError("profiling must be a mapping")
    if set(options) - {"level", "epochs", "models", "seeds", "stages"}:
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
    token = ACTIVE.set(None)
    try:
        if selected(recorder, number):
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profile = torch.profiler.profile(
                activities=activities, profile_memory=True, record_shapes=True
            )
            profile.__enter__()
            ACTIVE.set((device, pending))
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
        if profile is not None:
            # Collection is explicitly instrumented: synchronize once at epoch completion.
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    for event, start, end in pending:
                        event["cuda_elapsed_ms"] = start.elapsed_time(end)
                        event["cuda_scope"] = "current-stream CUDA event interval; not kernel sum"
                profile.__exit__(None, None, None)
                path = recorder.path.with_name(
                    recorder.path.stem + "-epoch" + str(number) + ".trace.json"
                )
                profile.export_chrome_trace(str(path))
                trace = json.loads(path.read_text(encoding="utf8"))
                kernels = sum(e.get("cat") == "kernel" for e in trace.get("traceEvents", []))
                if device.type == "cuda" and not kernels:
                    recorder.notes.append(
                        "CUDA kernel tracing unavailable: exported trace has no kernel events"
                    )
                recorder.artifacts.append(
                    {
                        "path": str(path),
                        "epoch": number,
                        "kernel_events": kernels,
                        "format": "pytorch_chrome_trace",
                        "device": str(device),
                    }
                )
                # CUDA durations resolved after operations finished; journal amended explicitly.
                for event, _, _ in pending:
                    recorder.emit({**event, "record_type": "end"})
            except Exception as exc:
                recorder.notes.append("Profiler export/device interval unavailable: " + str(exc))


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
                start.record()
            try:
                yield
            finally:
                if end is not None:
                    end.record()
                    pending.append((event, start, end))
                    event["gpu_allocated_bytes"] = torch.cuda.memory_allocated(device)
                    event["gpu_reserved_bytes"] = torch.cuda.memory_reserved(device)
