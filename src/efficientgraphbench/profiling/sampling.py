"""Bounded, optional memory observations; never reset allocator peaks or synchronize."""

import sys
import threading
import time
from contextlib import contextmanager


@contextmanager
def sample_memory(recorder, device, scope):
    options = recorder.options.get("memory_sampling") if recorder else None
    if options is None or getattr(recorder, "sampling_active", False):
        yield
        return
    if any(
        key in recorder.options and recorder.identity.get(field) not in recorder.options[key]
        for key, field in [("models", "model"), ("seeds", "seed"), ("stages", "stage")]
    ):
        yield
        return
    import psutil

    stop = threading.Event()
    process = psutil.Process()
    maximum = options.get("max_samples", 1000)
    interval = options.get("interval_ms", 50) / 1000
    recorder.sampling_active = True

    capture_lock = threading.Lock()

    def _capture(boundary, reserve_final=True):
        if len(recorder.memory_samples) >= maximum - int(reserve_final):
            return
        row = dict(
            record_type="memory_sample",
            schema_version=1,
            clock_domain=recorder.domain,
            offset_ns=time.perf_counter_ns() - recorder.origin,
            scope=scope,
            boundary=boundary,
            measurement="sampled point; not continuous history or an exact peak",
        )
        try:
            row["cpu_rss_bytes"] = process.memory_info().rss
            torch = sys.modules.get("torch")
            if torch and str(device).startswith("cuda") and torch.cuda.is_initialized():
                row["gpu_allocated_bytes"] = torch.cuda.memory_allocated(device)
                row["gpu_reserved_bytes"] = torch.cuda.memory_reserved(device)
                row["gpu_source"] = "PyTorch allocator query at sample time"
        except Exception as exc:
            row["unavailable_reason"] = str(exc)
        recorder.memory_samples.append(row)
        recorder.emit(row)

    def capture(boundary, reserve_final=True):
        with capture_lock:
            _capture(boundary, reserve_final)

    recorder.memory_failure_snapshot = lambda: capture("failure", reserve_final=False)

    def poll():
        while not stop.wait(interval):
            capture("interval")
            if len(recorder.memory_samples) >= maximum - 1:
                break

    capture("start")
    worker = threading.Thread(target=poll, daemon=True, name="egbench-memory-samples")
    worker.start()
    boundary = "end"
    try:
        yield
    except BaseException:
        boundary = "failure"
        raise
    finally:
        stop.set()
        worker.join()
        capture(boundary, reserve_final=False)
        recorder.sampling_active = False
        recorder.memory_failure_snapshot = None
        if len(recorder.memory_samples) >= maximum:
            recorder.notes.append("Memory sampling reached its per-process sample limit")
