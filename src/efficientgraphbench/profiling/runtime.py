import statistics
import threading
import time

import psutil
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class MemoryProfiler:
    """RSS sampled every 10 ms; CUDA allocator peak, not total device usage."""

    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.peak_cpu_bytes = 0
        self.peak_gpu_bytes = None
        self.peak_gpu_reserved_bytes = None
        self.stop = threading.Event()

    def sample(self):
        self.peak_cpu_bytes = max(self.peak_cpu_bytes, self.process.memory_info().rss)

    def _poll(self):
        while not self.stop.wait(0.01):
            self.sample()

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()
        if self.device.type == "cuda":
            self.peak_gpu_bytes = torch.cuda.max_memory_allocated(self.device)
            self.peak_gpu_reserved_bytes = torch.cuda.max_memory_reserved(self.device)

    def metrics(self):
        return {
            "peak_gpu_allocated_mib": (
                self.peak_gpu_bytes / 1024**2 if self.peak_gpu_bytes is not None else None
            ),
            "peak_gpu_reserved_mib": (
                self.peak_gpu_reserved_bytes / 1024**2
                if self.peak_gpu_reserved_bytes is not None
                else None
            ),
            "peak_cpu_memory_mb": self.peak_cpu_bytes / 1024**2,
            "peak_gpu_memory_mb": (
                self.peak_gpu_bytes / 1024**2 if self.peak_gpu_bytes is not None else None
            ),
        }


@torch.inference_mode()
def measure_latency(model, data, device, warmup, repeats):
    model.eval()
    for _ in range(warmup):
        model(data)
    timings = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        model(data)
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "inference_latency_mean_ms": statistics.mean(timings),
        "inference_latency_median_ms": statistics.median(timings),
        "inference_latency_std_ms": statistics.pstdev(timings),
        "inference_ms": statistics.median(timings),
        "inference_mean_ms": statistics.mean(timings),
    }
