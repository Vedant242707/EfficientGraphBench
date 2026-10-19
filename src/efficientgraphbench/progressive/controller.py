"""Stage barrier controller: every seed finishes before validation-only promotion."""

import gc
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
import torch_geometric

from efficientgraphbench.api.benchmark import make_config
from efficientgraphbench.benchmark.execution import RunnerError
from efficientgraphbench.benchmark.identity import physical_hardware
from efficientgraphbench.benchmark.interchange import file_sha256
from efficientgraphbench.benchmark.runner import resolve_device
from efficientgraphbench.datasets.registry import DatasetBundle, load_dataset
from efficientgraphbench.models.catalog import PRIMARY_MODELS, model_spec
from efficientgraphbench.progressive.baseline import compute_accounting, load_full_baseline
from efficientgraphbench.progressive.capabilities import progressive_capabilities
from efficientgraphbench.progressive.config import ProgressiveConfig
from efficientgraphbench.progressive.graphs import ProgressionBuilder
from efficientgraphbench.progressive.reference import train_reference_stage
from efficientgraphbench.progressive.results import ProgressiveResults
from efficientgraphbench.progressive.selection import select_candidates
from efficientgraphbench.progressive.training import train_stage


class ProgressiveBenchmark:
    def __init__(
        self, dataset="cora", models="all", seeds=None, runs=None, progressive=None, **options
    ):
        self.dataset = dataset
        self.models = (
            list(PRIMARY_MODELS)
            if models == "all"
            else ([models] if isinstance(models, str) else list(models))
        )
        self.models = list(dict.fromkeys(m.lower() for m in self.models))
        if runs is not None and (type(runs) is not int or runs < 1 or seeds is not None):
            raise ValueError("Use positive runs or explicit seeds, not both")
        self.seeds = list(dict.fromkeys(seeds if seeds is not None else range(1, (runs or 3) + 1)))
        if not self.models or not self.seeds:
            raise ValueError("Models and seeds cannot be empty")
        self.progressive = (
            progressive
            if isinstance(progressive, ProgressiveConfig)
            else ProgressiveConfig(**(progressive or {}))
        ).validate()
        self.options = options
        # Validate all candidates before allocating graphs or training any model.
        self.configs = {}
        for model in self.models:
            model_spec(model)
            for seed in self.seeds:
                cfg = make_config(
                    dataset if isinstance(dataset, str) else "cora", model, seed=seed, **options
                ).validate()
                if isinstance(dataset, DatasetBundle):
                    cfg.dataset = "custom"
                cfg.training.epochs = sum(s.additional_epochs for s in self.progressive.stages)
                if cfg.training.early_stopping_patience is not None:
                    raise ValueError(
                        "Progressive v1 uses exact additional budgets; disable early stopping"
                    )
                self.configs[model, seed] = cfg

    def run(self) -> ProgressiveResults:
        started = time.perf_counter()
        base = self.configs[self.models[0], self.seeds[0]]
        output = Path(base.output_dir) / ("progressive-" + uuid.uuid4().hex[:12])
        output.mkdir(parents=True, exist_ok=True)
        device = resolve_device(base.device)
        hardware, hardware_hash = physical_hardware(device)
        package = Path(__file__).resolve().parents[1]
        provenance = {
            "torch": str(torch.__version__),
            "pyg": torch_geometric.__version__,
            "source_snapshot": {
                str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
            },
        }
        bundle = (
            self.dataset
            if isinstance(self.dataset, DatasetBundle)
            else load_dataset(
                self.dataset,
                base.data_dir,
                dataset_path=base.dataset_path,
                split_seed=base.split_seed,
                split_index=base.split_index,
                split_mode=base.split_mode,
            )
        )
        if (
            base.dataset == "ogbn-arxiv"
            and isinstance(self.dataset, str)
            and self.progressive.progression_strategy == "auto"
        ):
            raise ValueError(
                "The current OGB loader omits temporal edge availability. "
                "Choose deterministic_nested explicitly for a non-temporal simulation, "
                "or supply a DatasetBundle with verified temporal metadata."
            )
        builder = ProgressionBuilder(bundle, self.progressive, output / "graphs")
        records, parents = [], {}
        active = list(self.models)
        metadata = {
            "benchmark_mode": "progressive_multi_fidelity",
            "config": self.progressive.to_dict(),
            "hardware": hardware,
            "hardware_fingerprint": hardware_hash,
            "software": provenance,
            "compute": {},
            "evaluation_semantics": "expanded_transductive_retraining",
        }
        recipes = {m: self.configs[m, self.seeds[0]].to_dict() for m in self.models}
        for recipe in recipes.values():
            recipe["training"]["epochs"] = sum(s.additional_epochs for s in self.progressive.stages)
        expected = {
            **builder.source,
            "models": self.models,
            "seeds": self.seeds,
            "hardware_fingerprint": hardware_hash,
            "source_snapshot": provenance["source_snapshot"],
            "software": provenance,
            "configs": recipes,
            "sources": {m: model_spec(m)["source_type"] for m in self.models},
            "commits": {m: model_spec(m).get("commit") for m in self.models},
        }
        if self.progressive.full_baseline:
            from efficientgraphbench.benchmark.execution_identity import execution_identity

            expected["execution_identities"] = {}
            for (model, seed), cfg in self.configs.items():
                recipe = cfg.to_dict()
                recipe["training"]["epochs"] = sum(
                    s.additional_epochs for s in self.progressive.stages
                )
                try:
                    expected["execution_identities"][model, seed] = execution_identity(
                        recipe, builder.source, hardware, model_spec(model)
                    )["fingerprint"]
                except (OSError, RuntimeError, ValueError):
                    expected["execution_identities"][model, seed] = None
        baseline = load_full_baseline(self.progressive.full_baseline, expected)
        metadata.update(
            schema_version=2,
            dataset=base.dataset if isinstance(self.dataset, str) else bundle.name,
            dataset_scale={
                "nodes": bundle.data.num_nodes,
                "edges": bundle.data.num_edges,
                "features": bundle.data.num_node_features,
                "classes": bundle.num_classes,
            },
            progression_strategy=builder.strategy,
            stage_node_counts=builder.counts,
            seeds=self.seeds,
            models=self.models,
            baseline=baseline,
            checkpoint_policy={
                "model": "resume",
                "optimizer": "resume",
                "scheduler": "resume when configured",
                "selection": "last stage weights; no best-validation restoration",
                "validation": "stage end and each recovery-tracking epoch until first crossing",
                "full_comparison": (
                    "Full uses per-epoch validation and best-validation weights; "
                    "equal epochs do not imply equal graph exposure"
                ),
            },
        )
        results = ProgressiveResults(records, output, metadata)
        for index, stage in enumerate(self.progressive.stages):
            build_start = time.perf_counter()
            graph = builder.build(index)
            graph_seconds = time.perf_counter() - build_start
            stage_records = []
            for model in active:
                failed_reason = None
                for seed in self.seeds:
                    cfg = self.configs[model, seed]
                    caps = progressive_capabilities(model)
                    row = {
                        "benchmark_mode": "progressive_multi_fidelity",
                        "dataset": base.dataset if isinstance(self.dataset, str) else bundle.name,
                        "model": model,
                        "seed": seed,
                        "hardware": hardware,
                        "hardware_fingerprint": hardware_hash,
                        "capabilities": caps,
                        "implementation": model_spec(model),
                        "num_nodes": graph.data.num_nodes,
                        "num_edges": graph.data.num_edges,
                        "new_nodes": graph.metadata["new_nodes"],
                        "fidelity": {
                            **graph.metadata,
                            "stage_index": index,
                            "stage_name": stage.name,
                            **asdict(stage),
                        },
                    }
                    interrupted = False
                    try:
                        if not caps["supports_progressive_resume"]:
                            row.update(
                                status="UNSUPPORTED_TASK",
                                candidate_status="unsupported",
                                failure_reason="progressive_resume_unsupported",
                            )
                        elif failed_reason:
                            row.update(
                                status="SKIPPED_DETERMINISTIC_FAILURE",
                                candidate_status="skipped",
                                failure_reason=failed_reason,
                            )
                        else:
                            attempt_start = time.perf_counter()
                            trainer = (
                                train_reference_stage
                                if caps["runner"] == "repository"
                                else train_stage
                            )
                            extra = (
                                {"hardware_hash": hardware_hash}
                                if caps["runner"] == "repository"
                                else {}
                            )
                            measured = trainer(
                                cfg,
                                self.progressive,
                                stage,
                                graph,
                                index,
                                bundle.num_classes,
                                device,
                                output,
                                provenance,
                                parents.get((model, seed)),
                                **extra,
                            )
                            row.update(measured)
                            if measured.get("status", "SUCCESS") != "SUCCESS":
                                error = RunnerError(
                                    measured["status"],
                                    measured.get("failure_reason", "Worker failed"),
                                )
                                error.progressive_measurements = measured
                                raise error
                            row.update(status="SUCCESS", candidate_status="advanced")
                            row["fidelity"].update(
                                checkpoint_id=measured["checkpoint_id"],
                                parent_checkpoint_id=measured["parent_checkpoint_id"],
                            )
                            parents[model, seed] = measured
                    except (
                        RuntimeError,
                        ValueError,
                        OSError,
                        ImportError,
                        RunnerError,
                        KeyboardInterrupt,
                    ) as exc:
                        oom = (
                            isinstance(exc, torch.OutOfMemoryError)
                            or "out of memory" in str(exc).lower()
                        )
                        status = (
                            "OOM"
                            if oom
                            else (
                                "CONFIG_ERROR" if isinstance(exc, ValueError) else "RUNTIME_ERROR"
                            )
                        )
                        interrupted = isinstance(exc, KeyboardInterrupt)
                        status = getattr(exc, "status", status)
                        row.update(getattr(exc, "progressive_measurements", {}))
                        failed_reason = f"{type(exc).__name__}: {exc}"
                        row.update(
                            status=status,
                            candidate_status="failed",
                            failure_reason=failed_reason,
                            failed_attempt_elapsed_seconds=time.perf_counter() - attempt_start,
                            failure_stage="progressive_training",
                            implementation_limit=None,
                        )
                    stage_records.append(row)
                    records.append(row)
                    if interrupted:
                        metadata["compute"] = compute_accounting(
                            sum(r.get("training_time_sec", 0) for r in records),
                            time.perf_counter() - started,
                            baseline,
                            records,
                        )
                        results.save()
                        raise KeyboardInterrupt("Interrupted attempt saved")
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            decisions = select_candidates(
                stage_records,
                stage,
                self.progressive.min_candidates,
                self.progressive.promotion_margin,
                self.progressive.uncertainty_aware,
                expected_seeds=self.seeds,
            )
            counts = {
                "nominal_candidates_to_retain": next(iter(decisions.values()), {}).get(
                    "nominal_retain_count", 0
                )
            }
            if index == len(self.progressive.stages) - 1:
                decisions = {
                    model: {
                        **decision,
                        "candidate_status": "advanced",
                        "reason": "completed_full_graph",
                    }
                    for model, decision in decisions.items()
                }
                counts["nominal_candidates_to_retain"] = len(decisions)
            for reason, label in [
                ("within_stage_cutoff", "rank"),
                ("within_promotion_margin", "margin"),
                ("uncertainty_overlap", "uncertainty"),
            ]:
                counts["candidates_retained_by_" + label] = sum(
                    d["reason"] == reason for d in decisions.values()
                )
            counts["actual_candidates_promoted"] = sum(
                d["candidate_status"] == "advanced" for d in decisions.values()
            )
            if index == len(self.progressive.stages) - 1:
                counts["candidates_retained_by_rank"] = len(decisions)
            metadata.setdefault("selection_counts", []).append(counts)
            for row in stage_records:
                if row["model"] in decisions:
                    decision = decisions[row["model"]]
                    if index == len(self.progressive.stages) - 1:
                        decision = {
                            **decision,
                            "candidate_status": "advanced",
                            "reason": "completed_full_graph",
                        }
                    row.update(candidate_status=decision["candidate_status"], promotion=decision)
                    if decision["candidate_status"] == "eliminated":
                        row["elimination"] = {
                            **decision,
                            "after_stage": stage.name,
                            "graph_fraction_reached": stage.graph_fraction,
                            "training_seconds_consumed_all_seeds": sum(
                                r.get("training_time_sec", 0)
                                for r in records
                                if r["model"] == row["model"]
                            ),
                        }
                elif row["status"] == "SUCCESS":
                    row.update(
                        candidate_status="failed",
                        failure_reason="Another seed failed; candidate cannot advance",
                    )
            active = [m for m, d in decisions.items() if d["candidate_status"] == "advanced"]
            metadata.setdefault("stage_graph_build_seconds", []).append(graph_seconds)
            elapsed = time.perf_counter() - started
            training = sum(r.get("training_time_sec", 0) for r in records)
            metadata["compute"] = compute_accounting(training, elapsed, baseline, records)
            results.save()
            if not active:
                break
        report_start = time.perf_counter()
        results.show_all(display=False)
        metadata["report_generation_seconds"] = time.perf_counter() - report_start
        metadata["compute"] = compute_accounting(
            sum(r.get("training_time_sec", 0) for r in records),
            time.perf_counter() - started,
            baseline,
            records,
        )
        metadata["compute"]["elapsed_scope"] = (
            "run_through_initial_report; final metadata refresh excluded"
        )
        results.show_all(display=False)
        return results
