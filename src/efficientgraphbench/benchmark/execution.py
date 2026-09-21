"""Execution boundaries; workers never need to import the controller package."""

import json
import math
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import psutil

STATUSES = {
    "RESOURCE_LIMIT",
    "ENVIRONMENT_NOT_INSTALLED",
    "SKIPPED_DETERMINISTIC_FAILURE",
    "SKIPPED_AFTER_OOM",
    "SUCCESS",
    "OOM",
    "UNSUPPORTED_TASK",
    "UNSUPPORTED_GRAPH_SIZE",
    "DEPENDENCY_ERROR",
    "CONFIG_ERROR",
    "PREPROCESSING_ERROR",
    "RUNTIME_ERROR",
}


class RunnerError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class BaseModelRunner(ABC):
    @abstractmethod
    def run(self, request):
        """Return a result, or raise a classified RunnerError."""


class InProcessRunner(BaseModelRunner):
    def __init__(self, callback):
        self.callback = callback

    def run(self, request):
        return self.callback(request)


class ExternalEnvironmentRunner(BaseModelRunner):
    def __init__(self, command, directory, timeout=7200):
        self.command = list(command)
        self.directory = Path(directory)
        self.timeout = timeout

    def run(self, request):
        self.directory.mkdir(parents=True, exist_ok=True)
        request_path = self.directory / "request.json"
        result_path = self.directory / "result.json"
        if result_path.exists() or request_path.exists():
            raise RunnerError("CONFIG_ERROR", "Refusing to reuse an existing worker exchange")
        request = {**request, "exchange_schema_version": 1}
        request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        try:
            environment = os.environ.copy()
            # Newer-controller allocator options are not understood by torch 1.13.
            environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            environment.pop("PYTORCH_ALLOC_CONF", None)
            threads = str(request.get("config", {}).get("threads", 4))
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[key] = threads
            with (self.directory / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [*self.command, str(request_path.resolve()), str(result_path.resolve())],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    env=environment,
                )
                try:
                    process.wait(timeout=self.timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    # Windows venv launcher may spawn the actual Python worker.
                    # Kill the complete owned tree, not only its launcher.
                    try:
                        children = psutil.Process(process.pid).children(recursive=True)
                    except psutil.NoSuchProcess:
                        children = []
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    process.wait(timeout=10)
                    psutil.wait_procs(children, timeout=10)
                    raise
        except FileNotFoundError as exc:
            raise RunnerError("DEPENDENCY_ERROR", str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerError("RESOURCE_LIMIT", f"Worker exceeded {self.timeout}s timeout") from exc
        if not result_path.exists():
            raise RunnerError(
                "RUNTIME_ERROR",
                f"Worker exited {process.returncode} without a result; see worker.log",
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("exchange_schema_version") != 1:
                raise ValueError("Unsupported exchange schema version")
            if result["status"] not in STATUSES:
                raise ValueError("Unknown worker status")
            for key in ("run_id", "model_id", "seed", "split_hash", "canonical_graph_hash"):
                if result.get(key) != request.get(key):
                    raise ValueError(f"Worker identity mismatch: {key}")
            if result["status"] == "SUCCESS":
                if process.returncode != 0:
                    raise ValueError("Worker reported success but exited unsuccessfully")
                for key in ("accuracy", "validation_accuracy"):
                    if not isinstance(result.get(key), (int, float)) or not 0 <= result[key] <= 1:
                        raise ValueError(f"Invalid worker {key}")
                if result.get("hardware_fingerprint") != request["hardware_fingerprint"]:
                    raise ValueError("Worker used different physical hardware")
                for key in (
                    "preprocessing_time_sec",
                    "training_time_sec",
                    "inference_latency_mean_ms",
                    "inference_latency_median_ms",
                    "inference_latency_std_ms",
                    "parameter_count",
                    "trainable_parameter_count",
                    "epochs_run",
                ):
                    value = result.get(key)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Missing or invalid worker metric: {key}")
                for key in ("peak_gpu_allocated_mib", "peak_gpu_reserved_mib"):
                    value = result.get(key)
                    if request.get("device") == "cuda" and (
                        type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    ):
                        raise ValueError(f"Missing or invalid worker GPU metric: {key}")
            return result
        except (ValueError, KeyError, TypeError) as exc:
            raise RunnerError("RUNTIME_ERROR", f"Invalid worker result: {exc}") from exc


class RepositoryRunner(ExternalEnvironmentRunner):
    def __init__(self, command, directory, repository, commit, timeout=7200):
        super().__init__(command, directory, timeout)
        self.repository, self.commit = Path(repository), commit

    def run(self, request):
        if not self.repository.is_dir():
            raise RunnerError("DEPENDENCY_ERROR", f"Missing repository: {self.repository}")
        try:
            base = [
                "git",
                "-c",
                f"safe.directory={self.repository.resolve().as_posix()}",
                "-C",
                str(self.repository),
            ]
            actual = subprocess.check_output([*base, "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(
                [*base, "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RunnerError("DEPENDENCY_ERROR", f"Cannot verify repository: {exc}") from exc
        if actual != self.commit or dirty:
            raise RunnerError("CONFIG_ERROR", "Reference checkout differs from pinned clean commit")
        return super().run({**request, "repository_commit": actual})
