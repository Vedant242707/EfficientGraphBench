"""Nested node prefixes, stable masks and explicitly bounded temporal support."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from efficientgraphbench.benchmark.interchange import export_graph, fingerprint
from efficientgraphbench.datasets.registry import DatasetBundle
from efficientgraphbench.datasets.validation import validate_graph
from efficientgraphbench.profiling.events import observed
from efficientgraphbench.progressive.config import ProgressiveConfig


@dataclass
class GraphStage:
    data: Data
    node_ids: torch.Tensor
    old_mask: torch.Tensor
    metadata: dict


class ProgressionBuilder:
    def __init__(self, bundle: DatasetBundle, config: ProgressiveConfig, output: Path):
        self.bundle, self.config, self.output = copy.copy(bundle), config.validate(), output
        data = bundle.data.clone().cpu()
        self.bundle.data = data
        validate_graph(data, bundle.num_classes)
        self.source = export_graph(self.bundle, output / "source")
        times = getattr(data, "node_time", None)
        if times is None:
            times = getattr(data, "node_year", None)
        self.strategy = config.progression_strategy
        if self.strategy == "auto":
            self.strategy = "temporal" if times is not None else "deterministic_nested"
        self.times = times.flatten() if times is not None else None
        self.edge_times = getattr(data, "edge_time", None)
        if self.strategy == "temporal":
            if times is None or not config.temporal_static_features or self.edge_times is None:
                raise ValueError(
                    "Temporal progression requires node_time/node_year, edge_time, "
                    "and temporal_static_features=True confirming features and labels "
                    "are available at node arrival; snapshots are not inferred"
                )
            if self.times.numel() != data.num_nodes or not torch.isfinite(self.times).all():
                raise ValueError("Invalid node timestamps")
            self.edge_times = self.edge_times.flatten()
            if (
                self.edge_times.numel() != data.num_edges
                or not torch.isfinite(self.edge_times).all()
            ):
                raise ValueError("Invalid edge timestamps")
            if bool((self.edge_times > self.times.max()).any()):
                raise ValueError("Edges after the final node arrival require explicit snapshots")
            self.order = torch.argsort(self.times, stable=True)
        else:
            generator = torch.Generator().manual_seed(config.progression_seed)
            self.order = torch.randperm(data.num_nodes, generator=generator)
        # No test/validation labels are consulted to construct the ordering.
        self.source_hash = fingerprint(
            {
                "graph": self.source["canonical_graph_hash"],
                "node_time": self.times.tolist() if self.times is not None else None,
                "edge_time": self.edge_times.tolist() if self.edge_times is not None else None,
            }
        )
        self.previous_count, self.parent_hash = 0, None
        self.next_index = 0
        self.counts = [math.ceil(s.graph_fraction * data.num_nodes) for s in config.stages]
        if len(set(self.counts)) != len(self.counts):
            raise ValueError("Stage fractions round to identical node counts; use fewer stages")
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.order, output / "node_order.pt")
        if self.strategy == "temporal":
            import numpy as np

            np.savez(
                output / "temporal_metadata.npz",
                node_time=self.times.numpy(),
                edge_time=self.edge_times.numpy(),
            )

    @observed("graph_stage_construction")
    def build(self, index: int) -> GraphStage:
        if index != self.next_index:
            raise ValueError("Graph stages must be built sequentially from stage zero")
        self.next_index += 1
        data = self.bundle.data
        count = self.counts[index]
        ids = self.order[:count]
        edges = data.edge_index
        edge_filter = torch.ones(data.num_edges, dtype=torch.bool)
        cutoff = None
        if self.strategy == "temporal":
            cutoff = self.times[ids].max().item()
            edge_filter &= self.edge_times <= cutoff
        selected_edges, _, included = subgraph(
            ids,
            edges[:, edge_filter],
            relabel_nodes=True,
            num_nodes=data.num_nodes,
            return_edge_mask=True,
        )
        stage = Data(x=data.x[ids].clone(), y=data.y[ids].clone(), edge_index=selected_edges)
        if getattr(data, "raw_x", None) is not None:
            stage.raw_x = data.raw_x[ids].clone()
        for key in ("train_mask", "val_mask", "test_mask"):
            setattr(stage, key, getattr(data, key)[ids].clone())
        for key in ("edge_attr", "edge_weight"):
            value = getattr(data, key, None)
            if value is not None:
                setattr(stage, key, value[edge_filter][included].clone())
        directory = self.output / f"stage-{index + 1}"
        exported = export_graph(
            DatasetBundle(
                stage,
                self.bundle.num_classes,
                split=self.bundle.split,
                preprocessing=self.bundle.preprocessing,
                edge_policy=self.bundle.edge_policy,
            ),
            directory,
        )
        old = torch.arange(count) < self.previous_count
        metadata = {
            **exported,
            "source_graph_hash": self.source_hash,
            "stage_graph_hash": exported["canonical_graph_hash"],
            "stage_split_hash": exported["split_hash"],
            "parent_stage_graph_hash": self.parent_hash,
            "selected_node_ids": ids.tolist(),
            "selected_edge_count": stage.num_edges,
            "progression_strategy": self.strategy,
            "seed": self.config.progression_seed,
            "temporal_cutoff": cutoff,
            "new_nodes": count - self.previous_count,
            "evaluation_semantics": "expanded_transductive_retraining",
            "ordering_policy": "label-blind prefixes; original masks retained",
        }
        self.previous_count, self.parent_hash = count, metadata["stage_graph_hash"]
        return GraphStage(stage, ids, old, metadata)
