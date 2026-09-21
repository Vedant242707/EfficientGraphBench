"""Hardware-aware graph model benchmarking."""

from efficientgraphbench.version import __version__

__all__ = [
    "Benchmark",
    "ScalingBenchmark",
    "BenchmarkResult",
    "BenchmarkResults",
    "HardwareProfile",
    "__version__",
]


def __getattr__(name):
    if name in {"Benchmark", "ScalingBenchmark"}:
        from efficientgraphbench.api import benchmark

        return getattr(benchmark, name)
    if name in {"BenchmarkResult", "BenchmarkResults"}:
        from efficientgraphbench.api import results

        return getattr(results, name)
    if name == "HardwareProfile":
        from efficientgraphbench.profiling.hardware import HardwareProfile

        return HardwareProfile
    raise AttributeError(name)
