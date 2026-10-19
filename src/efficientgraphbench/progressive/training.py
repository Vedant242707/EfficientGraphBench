"""Stage training with last-epoch continuation and validation-only adaptation."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from efficientgraphbench.config import BenchmarkConfig
    from efficientgraphbench.progressive.graphs import GraphStage

import psutil
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from efficientgraphbench.benchmark.interchange import fingerprint
from efficientgraphbench.profiling.runtime import MemoryProfiler, synchronize
from efficientgraphbench.profiling.timing import timed
from efficientgraphbench.progressive.adaptation import RecoveryTracker, transition_metrics
from efficientgraphbench.progressive.checkpoints import CheckpointManager
from efficientgraphbench.progressive.config import FidelityStage, ProgressiveConfig
from efficientgraphbench.training.evaluator import classification_metrics
from efficientgraphbench.training.seed import seed_everything


def evaluate(model: torch.nn.Module, data: Data, mask: torch.Tensor, metric: str) -> float | None:
    if not bool(mask.any()):
        return None
    model.eval()
    with torch.inference_mode():
        return classification_metrics(model(data), data.y, mask)[metric.removeprefix("val_")]


def _train_stage(
    config: BenchmarkConfig,
    progressive: ProgressiveConfig,
    stage: FidelityStage,
    graph: GraphStage,
    index: int,
    classes: int,
    device: torch.device,
    output: Path,
    provenance: dict,
    parent: dict | None,
    timings: dict,
    factory=None,
) -> dict:
    if not graph.data.train_mask.any() or not graph.data.val_mask.any():
        raise ValueError(
            "Stage needs nonempty training and validation masks; increase initial size"
        )
    seed_everything(config.seed)
    torch.set_num_threads(config.threads)
    identity = {
        "model": config.model.name,
        "seed": config.seed,
        "model_config_hash": fingerprint(config.to_dict()),
        "source_graph_hash": graph.metadata["source_graph_hash"],
        "software_provenance": provenance,
        "progressive_config_hash": fingerprint(progressive.to_dict()),
    }
    baseline = parent["validation_metric"] if parent else None
    epoch_count = parent["current_epoch"] if parent else 0
    before_rss = psutil.Process().memory_info().rss / 1024**2
    gpu_baseline = torch.cuda.memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    started = time.perf_counter()
    with MemoryProfiler(device) as memory:
        data = graph.data.clone().to(device)
        old = graph.old_mask.to(device) & data.val_mask
        new = ~graph.old_mask.to(device) & data.val_mask
        if factory is None:
            from efficientgraphbench.models.registry import build_model

            model = build_model(config.model, data.num_node_features, classes).to(device)
            optimizer = getattr(torch.optim, config.training.optimizer)(
                model.parameters(),
                lr=config.training.lr,
                weight_decay=config.training.weight_decay,
            )
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, sum(s.additional_epochs for s in progressive.stages)
                )
                if config.training.scheduler == "cosine"
                else None
            )
            settings = {}
        else:
            model, optimizer, scheduler, settings = factory(data, device)
        resumed = {}
        if parent:
            with timed(timings, "checkpoint_restore_seconds", device):
                restored = CheckpointManager.restore(
                    Path(parent["checkpoint_path"]),
                    parent["checkpoint_id"],
                    {
                        **identity,
                        "stage_index": index - 1,
                        "stage_graph_hash": graph.metadata["parent_stage_graph_hash"],
                    },
                    model,
                    optimizer,
                    scheduler,
                )
                if (
                    restored["current_epoch"] != parent["current_epoch"]
                    or restored["validation_metric"] != parent["validation_metric"]
                ):
                    raise ValueError("Parent epoch/validation metadata differs from checkpoint")
                resumed = {
                    k: restored.get(k)
                    for k in ("saved_model_signature", "saved_optimizer_signature")
                }
        # New instances have no stale graph-dependent propagation caches.
        memory.mark("initialize_and_restore")
        with timed(timings, "pre_adaptation_evaluation_seconds", device):
            pre_old = evaluate(model, data, old, progressive.selection_metric) if parent else None
            pre_new = evaluate(model, data, new, progressive.selection_metric) if parent else None
        recovery = RecoveryTracker(baseline, pre_old, progressive.recovery_threshold)
        synchronize(device)
        training_start = time.perf_counter()
        for epoch in range(1, stage.additional_epochs + 1):
            with timed(timings, "optimization_seconds", device):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(data)
                loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                if settings.get("clip_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
                optimizer.step()
                if scheduler:
                    scheduler.step()
            if parent and not recovery.recovered and bool(old.any()):
                with timed(timings, "validation_seconds", device):
                    score = evaluate(model, data, old, progressive.selection_metric)
                synchronize(device)
                recovery.observe(epoch, score, time.perf_counter() - training_start)
        synchronize(device)
        training_seconds = time.perf_counter() - training_start
        memory.mark("training_and_recovery_validation")
        with timed(timings, "post_adaptation_evaluation_seconds", device):
            old_score = evaluate(model, data, old, progressive.selection_metric)
            new_score = evaluate(model, data, new, progressive.selection_metric)
            score = evaluate(model, data, data.val_mask, progressive.selection_metric)
        checkpoint = output / "checkpoints" / config.model.name / str(config.seed) / f"{index}.pt"
        metadata = {
            **identity,
            "stage_index": index,
            "stage_config": asdict(stage),
            "stage_graph_hash": graph.metadata["stage_graph_hash"],
            "stage_split_hash": graph.metadata["stage_split_hash"],
            "parent_checkpoint_id": parent["checkpoint_id"] if parent else None,
            "current_epoch": epoch_count + stage.additional_epochs,
            "model_configuration": config.to_dict(),
            "validation_metric": score,
            "checkpoint_policy": {
                "model_state": "resume",
                "optimizer_state": "resume",
                "scheduler_state": "resume" if scheduler else "not_applicable",
                "weights": "last_epoch_not_best_validation",
            },
        }
        with timed(timings, "checkpoint_save_seconds", device):
            digest = CheckpointManager.save(checkpoint, model, optimizer, scheduler, metadata)
        test = None
        if index == len(progressive.stages) - 1 and bool(data.test_mask.any()):
            model.eval()
            with timed(timings, "test_evaluation_seconds", device), torch.inference_mode():
                test = classification_metrics(model(data), data.y, data.test_mask)
            test["classification_protocol"].update(checkpoint="final_progressive_epoch")
        memory.mark("evaluation_and_checkpoint")
    metrics = memory.metrics()
    return {
        **metadata,
        "checkpoint_id": digest,
        "resume_evidence": resumed,
        "checkpoint_path": str(checkpoint.resolve()),
        "validation_metric": score,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "underlying_layer_classes": sorted(
            {type(m).__module__ + "." + type(m).__name__ for m in model.modules()}
        ),
        "training_class_counts": torch.bincount(
            data.y[data.train_mask], minlength=classes
        ).tolist(),
        "test_metrics": test,
        "training_time_sec": training_seconds,
        "total_training_phase_seconds": training_seconds,
        **timings,
        "evaluation_policy": {
            "checkpoint": "last_epoch",
            "validation": "each epoch until first recovery; pre/post every stage",
        },
        "reference_settings": settings,
        "stage_wall_time_sec": time.perf_counter() - started,
        "additional_epochs": stage.additional_epochs,
        "adaptation": {
            "metric": progressive.selection_metric,
            "evaluation_split": "validation",
            "old_node_count": int(old.sum()),
            "new_node_count": int(new.sum()),
            "old_node_metric": old_score,
            "new_node_metric": new_score,
            "overall_metric": score,
            **transition_metrics(baseline, pre_old, pre_new, old_score, new_score),
            **recovery.result(old_score),
        },
        **metrics,
        "baseline_cpu_rss_mib": before_rss,
        "additional_cpu_peak_above_baseline_mib": max(
            0, metrics["peak_cpu_memory_mb"] - before_rss
        ),
        "baseline_gpu_allocated_mib": gpu_baseline,
        "additional_gpu_peak_above_baseline_mib": max(
            0, metrics["peak_gpu_allocated_mib"] - gpu_baseline
        )
        if gpu_baseline is not None
        else None,
    }


def train_stage(*args, factory=None, **kwargs) -> dict:
    timings = {}
    start = time.perf_counter()
    try:
        return _train_stage(*args, timings=timings, factory=factory, **kwargs)
    except BaseException as exc:
        exc.progressive_measurements = {
            **getattr(exc, "resource_measurements", {}),
            **timings,
            "failed_attempt_elapsed_seconds": time.perf_counter() - start,
        }
        raise
