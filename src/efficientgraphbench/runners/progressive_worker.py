"""Shared continuation adapter for official models; upstream files remain untouched."""

import copy
from pathlib import Path
from types import SimpleNamespace


def execute_stage(request, data, device, factory, software):
    import torch

    from efficientgraphbench.progressive.config import ProgressiveConfig
    from efficientgraphbench.progressive.training import train_stage

    raw_config = copy.deepcopy(request["config"])
    config = SimpleNamespace(**raw_config)
    config.model = SimpleNamespace(**raw_config["model"])
    config.training = SimpleNamespace(**raw_config["training"])
    config.to_dict = lambda: copy.deepcopy(raw_config)
    progressive = ProgressiveConfig(**request["progressive"]).validate()
    index = request["stage_index"]
    old = torch.arange(data.num_nodes) < request["old_node_count"]
    graph = SimpleNamespace(data=data, old_mask=old, metadata=request["stage_metadata"])
    provenance = {
        **request["controller_provenance"],
        "worker_software": software,
        "implementation": request["model_id"],
        "commit": request["repository_commit"],
    }

    def initialize(stage_data, target):
        # Factories rebuild graph-derived inputs on CPU for this stage only.
        stage_data.cpu()
        stage_data.batch = torch.zeros(stage_data.num_nodes, dtype=torch.long)
        local_request = copy.deepcopy(request)
        local_request["config"]["training"]["epochs"] = sum(
            s.additional_epochs for s in progressive.stages
        )
        model, forward, optimizer, scheduler, settings = factory(local_request, stage_data, target)
        stage_data.to(target)
        stage_data.all_mask = torch.ones(stage_data.num_nodes, dtype=torch.bool, device=target)

        class Interface(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.reference = model

            def forward(self, ignored):
                return forward("all")

        settings["task_modifications"] = [
            v for v in settings["task_modifications"] if "best validation" not in v
        ]
        settings["task_modifications"].append(
            "Progressive last-epoch continuation; exact additional budgets"
        )
        settings["early_stopping_patience"] = None
        settings["preprocessing_stage_graph_hash"] = request["canonical_graph_hash"]
        settings["preprocessing_recomputed"] = True
        settings["implementation_class"] = type(model).__module__ + "." + type(model).__name__
        return Interface(), optimizer, scheduler, settings

    return train_stage(
        config,
        progressive,
        progressive.stages[index],
        graph,
        index,
        request["num_classes"],
        device,
        Path(request["output_dir"]),
        provenance,
        request.get("parent"),
        factory=initialize,
    )
