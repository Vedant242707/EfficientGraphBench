"""Small public interface around the existing benchmark engine."""

import copy
import json
import uuid
from pathlib import Path

from efficientgraphbench.api.results import BenchmarkResult, BenchmarkResults
from efficientgraphbench.benchmark.interchange import export_graph
from efficientgraphbench.benchmark.matrix import run_matrix
from efficientgraphbench.benchmark.runner import run_benchmark
from efficientgraphbench.benchmark.scaling import run_scaling
from efficientgraphbench.config import BenchmarkConfig, ModelConfig, TrainingConfig
from efficientgraphbench.datasets.catalog import OFFICIAL_SPLIT_COUNTS, canonical_dataset
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.datasets.validation import validate_graph
from efficientgraphbench.models.catalog import PRIMARY_MODELS


def make_config(dataset, model, **options):
    options = dict(options)
    training = options.pop("training", None)
    training = (
        copy.deepcopy(training)
        if isinstance(training, TrainingConfig)
        else TrainingConfig(**(training or {}))
    )
    for key in TrainingConfig.__dataclass_fields__:
        if key in options:
            setattr(training, key, options.pop(key))
    config = BenchmarkConfig(model=ModelConfig(name=model.lower()), training=training, **options)
    if isinstance(dataset, DatasetBundle):
        validate_graph(dataset.data, dataset.num_classes)
        export_dir = Path(config.output_dir) / "inputs" / uuid.uuid4().hex
        exported = export_graph(dataset, export_dir)
        config.dataset = "custom"
        config.dataset_path = exported["graph_path"]
    elif isinstance(dataset, str):
        config.dataset = dataset
    else:
        raise TypeError("dataset must be a built-in name or DatasetBundle from load_dataset")
    return config


class Benchmark:
    def __init__(
        self, dataset="cora", models="all", seeds=None, runs=None, splits="fixed", **options
    ):
        if runs is not None and (type(runs) is not int or runs < 1):
            raise ValueError("runs must be a positive integer")
        if runs is not None and seeds is not None:
            raise ValueError("Use runs or explicit seeds, not both")
        if splits not in {"fixed", "random", "official"}:
            raise ValueError("splits must be fixed, random, or official")
        if splits != "fixed" and any(
            key in options for key in ("split_seed", "split_index", "split_mode")
        ):
            raise ValueError("splits selects the split settings; remove explicit split options")
        self.splits = splits
        self.dataset, self.options = dataset, options
        self.models = (
            list(PRIMARY_MODELS)
            if models == "all"
            else ([models] if isinstance(models, str) else list(models))
        )
        self.models = list(dict.fromkeys(m.lower() for m in self.models))
        self.seeds = list(
            dict.fromkeys(
                seeds if seeds is not None else (range(1, runs + 1) if runs else (42, 43, 44))
            )
        )
        if not self.models or not self.seeds:
            raise ValueError("models and seeds must not be empty")

    def run(self):
        jobs = []
        base = make_config(self.dataset, self.models[0], **self.options)
        for model in self.models:
            for index, seed in enumerate(self.seeds):
                config = copy.deepcopy(base)
                config.model.name, config.seed = model, seed
                if self.splits == "random":
                    config.split_mode, config.split_seed = "random", seed
                elif self.splits == "official":
                    count = OFFICIAL_SPLIT_COUNTS.get(canonical_dataset(config.dataset), 1)
                    config.split_index = index % count
                jobs.append(config)
        results = BenchmarkResults(run_matrix(jobs), base.output_dir)
        if self.splits != "fixed":
            results.aggregate_splits = True
        return results

    @staticmethod
    def run_model(model, dataset="cora", **options):
        return BenchmarkResult(run_benchmark(make_config(dataset, model, **options)))


class ScalingBenchmark:
    def __init__(
        self,
        dataset="cora",
        model=None,
        models=None,
        sizes=None,
        step=None,
        seeds=(42, 43, 44),
        **options,
    ):
        if model and models:
            raise ValueError("Supply model or models, not both")
        self.models = (
            [model]
            if model
            else (
                list(PRIMARY_MODELS)
                if models is None or models == "all"
                else ([models] if isinstance(models, str) else list(models))
            )
        )
        if not self.models:
            raise ValueError("models must not be empty")
        self.config = make_config(dataset, self.models[0], **options)
        self.sizes, self.step, self.seeds = sizes or [], step, list(seeds)

    def run(self):
        path = run_scaling(self.config, self.sizes, self.models, self.seeds, step=self.step)
        summary = json.loads(path.with_name("scaling-summary.json").read_text(encoding="utf-8"))
        rows = []
        for model, outcome in summary["models"].items():
            measured = (
                outcome["last_success"]
                if outcome.get("first_oom")
                else outcome.get("last_executed")
            )
            measured = measured or outcome.get("first_oom")
            if measured:
                rows.append(
                    {
                        **measured["result"],
                        "model": model,
                        "num_nodes": measured["num_nodes"],
                        "oom_nodes": outcome["first_oom"]["num_nodes"]
                        if outcome.get("first_oom")
                        else None,
                    }
                )
        result = BenchmarkResults(rows, path.parent)
        result.all_results_path = path
        return result
