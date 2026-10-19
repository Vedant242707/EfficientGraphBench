"""Validated stage budgets. Every budget means additional epochs."""

import math
from dataclasses import asdict, dataclass, field


@dataclass
class FidelityStage:
    name: str
    graph_fraction: float
    additional_epochs: int
    retain_fraction: float = 1.0
    retain_top_k: int | None = None


@dataclass
class ProgressiveConfig:
    stages: list[FidelityStage] = field(default_factory=list)
    progression_strategy: str = "auto"
    progression_seed: int = 0
    selection_metric: str = "val_accuracy"
    min_candidates: int = 1
    recovery_threshold: float = 0.95
    temporal_static_features: bool = False
    full_baseline: str | None = None
    promotion_margin: float = 0.0
    uncertainty_aware: bool = True

    def validate(self) -> "ProgressiveConfig":
        self.stages = [FidelityStage(**s) if isinstance(s, dict) else s for s in self.stages]
        if not self.stages or self.stages[-1].graph_fraction != 1.0:
            raise ValueError("Progressive stages must end at graph_fraction=1.0")
        previous = 0.0
        names = set()
        for stage in self.stages:
            if not stage.name or stage.name in names:
                raise ValueError("Stage names must be nonempty and unique")
            names.add(stage.name)
            if not previous < stage.graph_fraction <= 1:
                raise ValueError("Graph fractions must strictly increase in (0,1]")
            previous = stage.graph_fraction
            if type(stage.additional_epochs) is not int or stage.additional_epochs < 1:
                raise ValueError("additional_epochs must be a positive integer")
            if not 0 < stage.retain_fraction <= 1:
                raise ValueError("retain_fraction must be in (0,1]")
            if stage.retain_top_k is not None and (
                type(stage.retain_top_k) is not int or stage.retain_top_k < 1
            ):
                raise ValueError("retain_top_k must be positive")
        if self.progression_strategy not in {"auto", "temporal", "deterministic_nested"}:
            raise ValueError("Unknown progression_strategy")
        if type(self.progression_seed) is not int or not 0 <= self.progression_seed < 2**32:
            raise ValueError("progression_seed must be an integer in [0,2**32)")
        if self.selection_metric not in {"val_accuracy", "val_f1_macro", "val_f1_micro"}:
            raise ValueError("Selection must use val_accuracy, val_f1_macro or val_f1_micro")
        if type(self.min_candidates) is not int or self.min_candidates < 1:
            raise ValueError("min_candidates must be positive")
        if not 0 < self.recovery_threshold <= 1:
            raise ValueError("recovery_threshold must be in (0,1]")
        if not math.isfinite(self.promotion_margin) or not 0 <= self.promotion_margin <= 1:
            raise ValueError("promotion_margin must be a finite fraction in [0,1]")
        if type(self.uncertainty_aware) is not bool:
            raise ValueError("uncertainty_aware must be boolean")
        return self

    def to_dict(self) -> dict:
        return asdict(self)
