"""Hardware-aware graph model benchmarking."""

from efficientgraphbench.version import __version__

__all__ = [
    "ProgressiveBenchmark",
    "ProgressiveConfig",
    "FidelityStage",
    "Benchmark",
    "ScalingBenchmark",
    "BenchmarkResult",
    "BenchmarkResults",
    "HardwareProfile",
    "__version__",
]


def __getattr__(name):
    if name == "ProgressiveBenchmark":
        from efficientgraphbench.progressive.controller import ProgressiveBenchmark

        return ProgressiveBenchmark
    if name in {"ProgressiveConfig", "FidelityStage"}:
        from efficientgraphbench.progressive import config

        return getattr(config, name)
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
