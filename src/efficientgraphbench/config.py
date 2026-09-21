"""Validated, serializable configuration shared by every model."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from efficientgraphbench.datasets.catalog import DATASETS, canonical_dataset
from efficientgraphbench.names import MODEL_NAMES


@dataclass
class ModelConfig:
    name: str = "gcn"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4
    propagation_steps: int = 10
    alpha: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 0.01
    weight_decay: float = 0.0005
    latency_warmup: int = 10
    latency_repeats: int = 50
    optimizer: str = "Adam"
    scheduler: str | None = None
    early_stopping_patience: int | None = None


@dataclass
class BenchmarkConfig:
    dataset: str = "cora"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"
    seed: int = 42
    threads: int = 4
    data_dir: str = "data"
    output_dir: str = "experiments"
    dataset_path: str | None = None
    split_seed: int = 0
    split_index: int = 0
    model_options: dict = field(default_factory=dict)
    worker_timeout_sec: int = 7200

    def validate(self):
        self.dataset = canonical_dataset(self.dataset)
        self.model.name = self.model.name.strip().lower()
        if not isinstance(self.model_options, dict):
            raise ValueError("model_options must be a mapping")
        if self.model.name in {"graphgps", "sgformer", "sgformer_reference"}:
            allowed = {"lr", "weight_decay", "layers", "hidden_dim", "heads", "dropout"}
            if self.model.name != "graphgps":
                allowed |= {
                    "trans_layers",
                    "trans_dropout",
                    "graph_weight",
                    "alpha",
                    "trans_weight_decay",
                    "use_bn",
                    "use_residual",
                    "use_weight",
                    "use_act",
                    "patience",
                }
            if set(self.model_options) - allowed:
                raise ValueError(
                    "Unknown reference model_options: "
                    + ", ".join(sorted(set(self.model_options) - allowed))
                )
            for key, value in self.model_options.items():
                if key.startswith("use_"):
                    if type(value) is not bool:
                        raise ValueError(f"{key} must be boolean")
                elif key in {"layers", "hidden_dim", "heads", "trans_layers", "patience"}:
                    if type(value) is not int or value < 1:
                        raise ValueError(f"{key} must be a positive integer")
                elif type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{key} must be finite numeric")
                elif key in {"dropout", "trans_dropout"} and not 0 <= value < 1:
                    raise ValueError(f"{key} must be in [0,1)")
                elif key in {"graph_weight", "alpha"} and not 0 <= value <= 1:
                    raise ValueError(f"{key} must be in [0,1]")
                elif key == "lr" and value <= 0:
                    raise ValueError("lr must be positive")
                elif key in {"weight_decay", "trans_weight_decay"} and value < 0:
                    raise ValueError(f"{key} must be nonnegative")
            if self.model.name == "graphgps" and (
                self.model_options.get("hidden_dim", 64) % self.model_options.get("heads", 4)
                or self.model_options.get("hidden_dim", 64) <= 4
            ):
                raise ValueError(
                    "GraphGPS width must exceed LapPE width4 and be divisible by heads"
                )
        if self.model.name not in {"graphgps", "sgformer", "sgformer_reference", "graphormer"}:
            for key, value in self.model_options.items():
                if key in {
                    "hidden_dim",
                    "num_layers",
                    "dropout",
                    "heads",
                    "propagation_steps",
                    "alpha",
                }:
                    setattr(self.model, key, value)
                elif key in {
                    "lr",
                    "weight_decay",
                    "optimizer",
                    "scheduler",
                    "early_stopping_patience",
                }:
                    setattr(self.training, key, value)
                else:
                    raise ValueError(f"Unknown model_options setting: {key}")
        integers = {
            "worker_timeout_sec": self.worker_timeout_sec,
            "split_seed": self.split_seed,
            "split_index": self.split_index,
            "seed": self.seed,
            "threads": self.threads,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "heads": self.model.heads,
            "propagation_steps": self.model.propagation_steps,
            "epochs": self.training.epochs,
            "latency_warmup": self.training.latency_warmup,
            "latency_repeats": self.training.latency_repeats,
        }
        for name, value in integers.items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        for name, value in {
            "dropout": self.model.dropout,
            "alpha": self.model.alpha,
            "lr": self.training.lr,
            "weight_decay": self.training.weight_decay,
        }.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.dataset not in DATASETS:
            raise ValueError("unsupported dataset; run egbench datasets to see available choices")
        if self.dataset == "custom" and not self.dataset_path:
            raise ValueError("custom data requires --dataset-path (CSV folder or NPZ file)")
        if self.dataset != "custom" and self.dataset_path:
            raise ValueError("--dataset-path is only used with --dataset custom")
        if not 0 <= self.split_seed < 2**32 or not 0 <= self.split_index < 20:
            raise ValueError("split_seed must be in [0, 2**32); split_index must be in [0, 20)")
        if self.dataset != "wikics" and self.split_index != 0:
            raise ValueError("split_index applies only to WikiCS")
        if self.model.name not in MODEL_NAMES:
            raise ValueError(f"model must be one of: {', '.join(MODEL_NAMES.values())}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        m, t = self.model, self.training
        if m.name == "appnp" and m.num_layers != 2:
            raise ValueError(
                "Reference APPNP has two feature layers; use appnp_adapted for other depths"
            )
        if t.optimizer not in {"Adam", "AdamW", "SGD"} or t.scheduler not in {None, "cosine"}:
            raise ValueError("optimizer must be Adam/AdamW/SGD; scheduler must be null/cosine")
        if t.early_stopping_patience is not None and (
            type(t.early_stopping_patience) is not int or t.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be null or a positive integer")
        if m.hidden_dim < 1 or m.num_layers < 2 or m.heads < 1:
            raise ValueError("hidden_dim/heads must be positive and num_layers >= 2")
        if (
            m.name in {"gat", "gatv2", "graphormer_adapted", "graphgps_adapted"}
            and m.hidden_dim % m.heads
        ):
            raise ValueError("Attention model hidden_dim must be divisible by heads")
        if m.propagation_steps < 1 or not 0 < m.alpha <= 1:
            raise ValueError("propagation_steps must be positive and alpha in (0, 1]")
        if not 0 <= m.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if t.epochs < 1 or not t.lr > 0 or not t.weight_decay >= 0:
            raise ValueError("epochs/lr must be positive and weight_decay nonnegative")
        if t.latency_warmup < 0 or t.latency_repeats < 1 or self.threads < 1:
            raise ValueError("invalid latency counts or CPU thread count")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not isinstance(self.model_options, dict) or self.worker_timeout_sec < 1:
            raise ValueError("model_options must be a mapping and worker_timeout_sec positive")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path: Path | None = None) -> BenchmarkConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    try:
        raw = dict(raw)
        raw["model"] = ModelConfig(**raw.get("model", {}))
        raw["training"] = TrainingConfig(**raw.get("training", {}))
        return BenchmarkConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc
