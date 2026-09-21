import gc
import hashlib
import json
import logging
import platform
import subprocess
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import psutil
import torch
import yaml

from efficientgraphbench.benchmark.execution import InProcessRunner, RepositoryRunner, RunnerError
from efficientgraphbench.benchmark.identity import physical_hardware
from efficientgraphbench.benchmark.interchange import export_graph, fingerprint
from efficientgraphbench.datasets.registry import load_dataset
from efficientgraphbench.models.catalog import ROOT, model_spec
from efficientgraphbench.models.graphormer import prepare_graphormer
from efficientgraphbench.models.registry import MODEL_METADATA, build_model
from efficientgraphbench.names import dataset_name, model_name
from efficientgraphbench.paths import CHECKOUT, PACKAGE, WORKER, environment_dir
from efficientgraphbench.profiling.hardware import hardware_info
from efficientgraphbench.profiling.runtime import MemoryProfiler, measure_latency, synchronize
from efficientgraphbench.results.status import UnsupportedGraphSize
from efficientgraphbench.results.writer import write_result
from efficientgraphbench.training.seed import seed_everything
from efficientgraphbench.training.trainer import train
from efficientgraphbench.version import __version__

logger = logging.getLogger(__name__)


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu or auto")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def run_benchmark(config):
    try:
        config.validate()
        resolve_device(config.device)
    except (ValueError, TypeError) as exc:
        record = {
            "schema_version": 4,
            "run_id": uuid.uuid4().hex,
            "batch_id": getattr(config, "_result_batch", None),
            "dataset": config.dataset,
            "model": config.model.name,
            "seed": config.seed,
            "status": "CONFIG_ERROR",
            "failure_stage": "configuration",
            "num_nodes": None,
            "num_edges": None,
            "failure_reason": str(exc),
            "error": str(exc),
            "implementation_limit": None,
            "graph_size_note": "Unavailable: configuration failed before dataset loading",
            "hardware": hardware_info(),
            "config": config.to_dict(),
            "comparison_group": "configuration-error",
            "device": config.device,
        }
        write_result(record, config.output_dir)
        return record
    return InProcessRunner(_run_validated).run(config)


def _run_validated(config):
    device = resolve_device(config.device)
    torch.set_num_threads(config.threads)
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=config.threads)
    seed_everything(config.seed)
    root = Path(config.output_dir)
    for directory in ("configs", "checkpoints", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    run_id = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{config.dataset}-"
        f"{config.model.name}-seed{config.seed}-{uuid.uuid4().hex[:8]}"
    )
    config_path = root / "configs" / f"{run_id}.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    comparison = config.to_dict()
    for key in ("seed", "output_dir", "data_dir", "model"):
        comparison.pop(key)
    comparison["model_settings"] = {
        k: v for k, v in config.to_dict()["model"].items() if k != "name"
    }
    hardware = {
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "pyg_version": version("torch-geometric"),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        hardware["gpu_memory_mb"] = hardware_info()["gpu_memory_mb"]
    comparison.update(hardware)
    comparison["protocol"] = "node-classification-transductive-fullbatch-v2"
    group_id = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()[:16]
    record = {
        "schema_version": 3,
        "batch_id": getattr(config, "_result_batch", None),
        "run_id": run_id,
        "dataset": config.dataset,
        "model": config.model.name,
        "model_display_name": model_name(config.model.name),
        "dataset_display_name": dataset_name(config.dataset),
        "seed": config.seed,
        "status": "RUNTIME_ERROR",
        "num_nodes": None,
        "num_edges": None,
        "failure_reason": None,
        "implementation_limit": None,
        "hardware": {
            **hardware,
            "cpu": platform.processor(),
            "ram_available_mb": psutil.virtual_memory().available / 1024**2,
            "gpu_free_memory_mb": hardware_info().get("gpu_free_memory_mb"),
        },
        "comparison_group": group_id,
        **hardware,
        "config": config.to_dict(),
        "model_metadata": MODEL_METADATA[config.model.name],
        "optimizer": "Adam",
        "batch_size": "full-graph",
        "scaling_context": getattr(config, "_scaling_context", None),
        "config_path": str(config_path),
    }
    handler = logging.FileHandler(root / "logs" / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    package_logger = logging.getLogger("efficientgraphbench")
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    profiler = MemoryProfiler(device)
    spec = model_spec(config.model.name)
    external = spec["runner"] == "repository"
    stage = "preprocessing"
    profiling_started = False
    try:
        physical, hardware_hash = physical_hardware(device)
        record.update(
            schema_version=4,
            model_id=config.model.name,
            display_name=model_name(config.model.name),
            implementation_type=spec["implementation_type"],
            source_type=spec["source_type"],
            category=spec["category"],
            environment=spec["environment"],
            repository_url=spec.get("repository_url"),
            repository_commit=spec.get("commit"),
            physical_hardware=physical,
            hardware_fingerprint=hardware_hash,
            software={
                "python": platform.python_version(),
                "torch": torch.__version__,
                "pyg": version("torch-geometric"),
                "cuda": torch.version.cuda,
            },
            profiler_settings={
                "warmup": config.training.latency_warmup,
                "repeats": config.training.latency_repeats,
                "scope": "warmed-full-graph-forward-including-interface",
                "gpu_memory": "torch-allocator-allocated-and-reserved",
                "cpu_memory": "process-RSS-sampled-10ms",
                "memory_scope": "preprocessing-model-initialization-training-inference",
                "cpu_threads": config.threads,
            },
        )
        env_dir = environment_dir(spec["environment"])
        record["environment_specification"] = {
            name: (env_dir / name).read_text(encoding="utf-8")
            for name in ("environment.yml", "metadata.json", "requirements-lock.txt")
            if (env_dir / name).exists()
        }
        try:
            record["efficientgraphbench_commit"] = subprocess.check_output(
                ["git", "-C", str(CHECKOUT or PACKAGE), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            record["efficientgraphbench_commit"] = None
        record["efficientgraphbench_version"] = __version__
        record["source_snapshot"] = {
            str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in PACKAGE.rglob("*.py")
        }
        if not external:
            profiler.__enter__()
            profiling_started = True
        logger.info("Starting %s on %s", run_id, hardware["device_name"])
        start = time.perf_counter()
        bundle = load_dataset(
            config.dataset,
            config.data_dir,
            dataset_path=config.dataset_path,
            split_seed=config.split_seed,
            split_index=config.split_index,
        )
        record.update(bundle.metadata())
        exchange_dir = root / "exchange" / run_id
        interchange = export_graph(bundle, exchange_dir)
        record.update(interchange)
        record["comparison_group"] = fingerprint(
            {
                "hardware": hardware_hash,
                "graph": interchange["canonical_graph_hash"],
                "split": interchange["split_hash"],
                "task": "transductive-node-classification",
                "metric": "accuracy",
                "profiler": record["profiler_settings"],
                "category": "experimental" if spec["category"] == "experimental" else "primary",
            }
        )[:16]
        if spec["runner"] == "unsupported":
            raise RunnerError("UNSUPPORTED_TASK", spec["reason"])
        if external:
            if not spec.get("python") or not (ROOT / spec["python"]).exists():
                raise RunnerError(
                    "ENVIRONMENT_NOT_INSTALLED", "Run egbench setup " + spec["environment"]
                )
            stage = "worker"
            request = {
                **interchange,
                "schema_version": 1,
                "run_id": run_id,
                "model_id": config.model.name,
                "seed": config.seed,
                "hardware_fingerprint": hardware_hash,
                "device": device.type,
                "config": config.to_dict(),
                "repository": str((ROOT / spec["repository"]).resolve()),
                "repository_commit": spec["commit"],
                "environment": spec,
                "controller_preprocessing_time_sec": time.perf_counter() - start,
            }
            worker = RepositoryRunner(
                [str(ROOT / spec["python"]), str(WORKER)],
                exchange_dir,
                ROOT / spec["repository"],
                spec["commit"],
                timeout=config.worker_timeout_sec,
            )
            record["controller_preprocessing_time_sec"] = request[
                "controller_preprocessing_time_sec"
            ]
            worker_start = time.perf_counter()
            try:
                record.update(worker.run(request))
            finally:
                record["worker_wall_time_sec"] = time.perf_counter() - worker_start
            return record
        if config.model.name == "graphormer_adapted":
            prepare_graphormer(bundle.data)
        comparison.update(
            {
                key: record[key]
                for key in (
                    "dataset_fingerprint",
                    "split",
                    "feature_preprocessing",
                    "edge_policy",
                )
            }
        )
        record["preprocessing_time_sec"] = time.perf_counter() - start
        stage = "allocation"
        with nullcontext():
            if getattr(bundle.data, "raw_x", None) is not None:
                del bundle.data.raw_x
            data = bundle.data.to(device)
            model = build_model(config.model, data.num_node_features, bundle.num_classes).to(device)
            import inspect

            implementation_file = Path(inspect.getfile(type(model.model)))
            record["implementation_class"] = (
                f"{type(model.model).__module__}.{type(model.model).__name__}"
            )
            record["implementation_source"] = str(implementation_file)
            record["implementation_source_sha256"] = hashlib.sha256(
                implementation_file.read_bytes()
            ).hexdigest()
            record["wrapper_class"] = "efficientgraphbench.models.registry.GraphModel"
            record["architecture_modified"] = spec["source_type"] not in {
                "library",
                "official_repo",
            }
            if config.model.name == "appnp":
                record["architecture_modifications"] = []
                record["task_modifications"] = [
                    "Execute unchanged upstream Net class without top-level CLI/training script",
                    "Canonical masks and best validation checkpoint replace the source harness",
                    "Explicit configured epoch budget and validation-accuracy patience; upstream "
                    "loss-window early stopping is not used",
                ]
            record["num_parameters"] = sum(p.numel() for p in model.parameters())
            record["parameter_count"] = record["num_parameters"]
            record["trainable_parameter_count"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            record["underlying_layer_classes"] = sorted(
                {f"{type(layer).__module__}.{type(layer).__name__}" for layer in model.modules()}
            )
            checkpoint = root / "checkpoints" / f"{run_id}.pt"
            stage = "training"
            record.update(train(model, data, config.training, device, checkpoint))
            record["checkpoint_path"] = str(checkpoint)
            record.update(
                measure_latency(
                    model,
                    data,
                    device,
                    config.training.latency_warmup,
                    config.training.latency_repeats,
                )
            )
            synchronize(device)
        record["status"] = "SUCCESS"
        logger.info("Completed accuracy=%.4f", record["accuracy"])
    except RunnerError as exc:
        record.update(
            status=exc.status, error=str(exc), failure_reason=str(exc), failure_stage=stage
        )
    except ImportError as exc:
        record.update(status="DEPENDENCY_ERROR", error=str(exc), failure_reason=str(exc))
    except UnsupportedGraphSize as exc:
        record.update(
            status="UNSUPPORTED_GRAPH_SIZE",
            error=str(exc),
            failure_reason=str(exc),
            implementation_limit=exc.limit,
        )
        logger.warning("Unsupported graph size: %s", exc)
    except (torch.OutOfMemoryError, MemoryError) as exc:
        record.update(status="OOM", error=str(exc), failure_reason=str(exc))
        logger.exception("Out of memory")
    except Exception as exc:
        record.update(
            status="PREPROCESSING_ERROR" if stage == "preprocessing" else "RUNTIME_ERROR",
            error=f"{type(exc).__name__}: {exc}",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        logger.exception("Benchmark failed")
    finally:
        if record.get("status") != "SUCCESS":
            record.setdefault("failure_stage", stage)
        if "preprocessing_time_sec" not in record and "start" in locals():
            record["elapsed_until_failure_sec"] = time.perf_counter() - start
            record["preprocessing_time_sec"] = None if external else time.perf_counter() - start
        if record["num_nodes"] is None:
            record["graph_size_note"] = "Unavailable: dataset loading did not complete"
        if not external:
            if profiling_started:
                profiler.__exit__(None, None, None)
            record.update(profiler.metrics())
        try:
            write_result(record, root)
        finally:
            package_logger.removeHandler(handler)
            handler.close()
            # Release this run before the next model, including failed allocations.
            model = data = bundle = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return record
