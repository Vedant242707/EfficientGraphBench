"""Standalone Python 3.10 worker. Reference source is imported, never rewritten."""

import hashlib
import json
import platform
import random
import sys
import time
import traceback
from importlib.metadata import distributions
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "profiling"))
sys.path.insert(0, str(ROOT / "benchmark"))


def graphgps(request, data, device):
    import graphgps  # noqa: F401 - registers official encoders, layers, heads and configuration
    import torch
    import yaml
    from graphgps.network.gps_model import GPSModel
    from graphgps.optimizer.extra_optimizers import cosine_with_warmup_scheduler
    from graphgps.transform.posenc_stats import compute_posenc_stats
    from torch_geometric.graphgym.config import cfg, set_cfg

    if (
        request["config"]["dataset"] in {"cora", "citeseer", "pubmed"}
        and getattr(data, "raw_x", None) is None
    ):
        raise ValueError("Raw source features missing; rerun with the current canonical exporter")
    if getattr(data, "raw_x", None) is not None:
        data.x = data.raw_x

    set_cfg(cfg)
    # Existing official transductive node recipe; no model implementation changes.
    recipe_path = Path(request["repository"]) / "configs/GPS/actor-GPS.yaml"
    cfg.merge_from_file(str(recipe_path))
    cfg.share.dim_in = data.num_node_features
    cfg.share.dim_out = request["num_classes"]
    cfg.dataset.name = request["config"]["dataset"]
    cfg.accelerator = str(device)
    cfg.wandb.use = False
    cfg.optim.max_epoch = request["config"]["training"]["epochs"]
    options = request["config"].get("model_options", {})
    allowed = {"lr", "weight_decay", "layers", "hidden_dim", "heads", "dropout"}
    if set(options) - allowed:
        raise ValueError(f"Unknown GraphGPS options: {set(options) - allowed}")
    for key, target, attr in (
        ("lr", cfg.optim, "base_lr"),
        ("weight_decay", cfg.optim, "weight_decay"),
        ("layers", cfg.gt, "layers"),
        ("heads", cfg.gt, "n_heads"),
        ("dropout", cfg.gt, "dropout"),
    ):
        if key in options:
            setattr(target, attr, options[key])
    if "hidden_dim" in options:
        cfg.gt.dim_hidden = cfg.gnn.dim_inner = options["hidden_dim"]
    compute_posenc_stats(data, ["LapPE"], data.is_undirected(), cfg)
    model = GPSModel(data.num_node_features, request["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = cosine_with_warmup_scheduler(
        optimizer, cfg.optim.num_warmup_epochs, cfg.optim.max_epoch
    )

    def forward(split):
        # Upstream mutates batch.x during encoding. Shallow Data copy preserves tensors
        # while preventing subsequent calls from re-encoding an already encoded x.
        import copy

        batch = copy.copy(data)
        batch.split = split
        return model(batch)[0]

    return (
        model,
        forward,
        optimizer,
        scheduler,
        {
            "recipe_source": str(recipe_path),
            "recipe": yaml.safe_load(cfg.dump()),
            "optimizer": "AdamW",
            "learning_rate": cfg.optim.base_lr,
            "weight_decay": cfg.optim.weight_decay,
            "scheduler": "cosine_with_warmup",
            "clip_grad_norm": 1.0,
            "early_stopping_patience": None,
            "architecture_modified": False,
            "architecture_modifications": [],
            "task_modifications": [
                "Official Actor node recipe applied to canonical input dataset",
                "Canonical split replaces upstream dataset-specific splits",
                "Raw Planetoid features, matching the upstream Planetoid loader",
                "Validation every epoch; best validation checkpoint selection",
            ],
            "preprocessing_requirements": ["official LapPE dense eigendecomposition"],
        },
    )


def sgformer(request, data, device):
    import torch
    from models import GCN
    from ours import SGFormer
    from torch_geometric.utils import to_undirected

    if (
        request["config"]["dataset"] in {"cora", "citeseer", "pubmed"}
        and getattr(data, "raw_x", None) is None
    ):
        raise ValueError("Raw source features missing; rerun with the current canonical exporter")
    if getattr(data, "raw_x", None) is not None:
        data.x = data.raw_x
    data.edge_index = to_undirected(data.edge_index)

    pubmed = request["config"]["dataset"] == "pubmed"
    options = dict(
        hidden_dim=64,
        layers=4,
        heads=1,
        dropout=0.5,
        trans_layers=1,
        trans_dropout=0.3 if pubmed else 0.2,
        graph_weight=0.8,
        alpha=0.5,
        lr=0.005 if pubmed else 0.01,
        weight_decay=0.0005,
        trans_weight_decay=0.01 if pubmed else 0.001,
        use_bn=False,
        use_residual=False,
        use_weight=False,
        use_act=False,
        patience=200,
    )
    overrides = request["config"].get("model_options", {})
    if set(overrides) - set(options):
        raise ValueError(f"Unknown SGFormer options: {set(overrides) - set(options)}")
    options.update(overrides)
    o = SimpleNamespace(**options)
    gnn = GCN(
        data.num_node_features,
        o.hidden_dim,
        o.hidden_dim,
        num_layers=o.layers,
        dropout=o.dropout,
        use_bn=o.use_bn,
    )
    model = SGFormer(
        data.num_node_features,
        o.hidden_dim,
        request["num_classes"],
        num_layers=o.trans_layers,
        num_heads=o.heads,
        alpha=o.alpha,
        dropout=o.trans_dropout,
        use_bn=o.use_bn,
        use_residual=o.use_residual,
        use_weight=o.use_weight,
        use_act=o.use_act,
        use_graph=True,
        graph_weight=o.graph_weight,
        gnn=gnn,
        aggregate="add",
    ).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": model.params1, "weight_decay": o.trans_weight_decay},
            {"params": model.params2, "weight_decay": o.weight_decay},
        ],
        lr=o.lr,
    )

    def forward(split):
        inputs = SimpleNamespace(graph={"node_feat": data.x, "edge_index": data.edge_index})
        return model(inputs)[getattr(data, f"{split}_mask")]

    return (
        model,
        forward,
        optimizer,
        None,
        {
            "recipe_source": str(Path(request["repository"]) / "medium/run.sh"),
            "recipe": options,
            "optimizer": "Adam",
            "learning_rate": o.lr,
            "weight_decay": {"transformer": o.trans_weight_decay, "graph_and_head": o.weight_decay},
            "scheduler": None,
            "clip_grad_norm": None,
            "early_stopping_patience": o.patience,
            "architecture_modified": False,
            "architecture_modifications": [],
            "task_modifications": [
                "Canonical split instead of upstream random class splits",
                "Raw source features retained, matching upstream --no_feat_norm",
                "Edges converted to undirected, matching upstream main.py",
                "run.sh --use_residual targets a different parser flag; actual "
                "ours_use_residual default False is preserved and recorded",
                "Common explicit epoch budget; source default is 500",
            ],
            "preprocessing_requirements": ["Canonical tensor conversion"],
        },
    )


def execute(request, result_path):
    result = {
        key: request[key]
        for key in ("run_id", "model_id", "seed", "split_hash", "canonical_graph_hash")
    }
    result.update(status="RUNTIME_ERROR", failure_reason=None, implementation_limit=None)
    result["exchange_schema_version"] = 1
    if request.get("exchange_schema_version") != 1:
        result.update(status="CONFIG_ERROR", failure_reason="Unsupported exchange schema version")
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return
    result["environment_overrides"] = {
        "PYTORCH_CUDA_ALLOC_CONF": None,
        "PYTORCH_ALLOC_CONF": None,
        "reason": "Controller allocator options may be unsupported by reference PyTorch",
    }
    stage, profiler = "dependencies", None
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torch_geometric
        from identity import physical_hardware
        from interchange import fingerprint
        from runtime import MemoryProfiler, measure_latency, synchronize
        from torch_geometric.data import Data

        device = torch.device(request["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ImportError("Reference environment cannot access CUDA")
        physical, hardware_hash = physical_hardware(device)
        result.update(
            physical_hardware=physical,
            hardware_fingerprint=hardware_hash,
            software={
                "python": platform.python_version(),
                "torch": torch.__version__,
                "pyg": torch_geometric.__version__,
                "cuda": torch.version.cuda,
            },
            torch_version=torch.__version__,
            pyg_version=torch_geometric.__version__,
            cuda_version=torch.version.cuda,
            dependencies={d.metadata["Name"]: d.version for d in distributions()},
        )
        if hardware_hash != request["hardware_fingerprint"]:
            raise ValueError("Worker hardware differs from controller")
        torch.set_num_threads(request["config"]["threads"])
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=request["config"]["threads"])
        random.seed(request["seed"])
        np.random.seed(request["seed"])
        torch.manual_seed(request["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(request["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        repository = Path(request["repository"])
        sys.path.insert(
            0, str(repository if request["model_id"] == "graphgps" else repository / "medium")
        )
        profiler = MemoryProfiler(device)
        with profiler:
            stage = "preprocessing"
            start = time.perf_counter()
            graph_path = Path(request["graph_path"])
            if hashlib.sha256(graph_path.read_bytes()).hexdigest() != request["graph_file_sha256"]:
                raise ValueError("Canonical graph file was modified")
            with np.load(graph_path, allow_pickle=False) as arrays:
                data = Data(**{key: torch.from_numpy(arrays[key].copy()) for key in arrays.files})
            splits = {
                k: getattr(data, f"{k}_mask").nonzero().flatten().tolist()
                for k in ("train", "val", "test")
            }
            if fingerprint(splits) != request["split_hash"]:
                raise ValueError("Split hash mismatch")
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            # Official preprocessing occurs on CPU before transfer.
            factory = graphgps if request["model_id"] == "graphgps" else sgformer
            model, forward, optimizer, scheduler, settings = factory(request, data, device)
            if getattr(data, "raw_x", None) is not None:
                del data.raw_x
            data = data.to(device)
            synchronize(device)
            result.update(settings)
            result["model_preprocessing_time_sec"] = time.perf_counter() - start
            result["preprocessing_time_sec"] = (
                result["model_preprocessing_time_sec"]
                + request["controller_preprocessing_time_sec"]
            )
            result["parameter_count"] = sum(p.numel() for p in model.parameters())
            result["trainable_parameter_count"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            result["num_parameters"] = result["parameter_count"]
            result["implementation_class"] = f"{type(model).__module__}.{type(model).__name__}"
            result["underlying_layer_classes"] = sorted(
                {f"{type(layer).__module__}.{type(layer).__name__}" for layer in model.modules()}
            )
            stage = "training"
            epochs = request["config"]["training"]["epochs"]
            best, best_state, best_epoch = -1, None, None
            stale_epochs = 0
            termination = "maximum_epochs"
            synchronize(device)
            start = time.perf_counter()
            optimization = 0.0
            for epoch in range(1, epochs + 1):
                step_start = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = forward("train")
                loss = F.cross_entropy(logits, data.y[data.train_mask])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                if settings["clip_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_grad_norm"])
                optimizer.step()
                synchronize(device)
                optimization += time.perf_counter() - step_start
                model.eval()
                with torch.inference_mode():
                    score = (
                        (forward("val").argmax(-1) == data.y[data.val_mask]).float().mean().item()
                    )
                if score > best:
                    best, best_epoch = score, epoch
                    stale_epochs = 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    stale_epochs += 1
                if scheduler is not None:
                    scheduler.step()
                if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                    print(
                        f"epoch={epoch} loss={loss.item():.4f} validation={score:.4f}", flush=True
                    )
                patience = settings["early_stopping_patience"]
                if patience is not None and stale_epochs >= patience:
                    termination = "early_stopping"
                    break
            synchronize(device)
            elapsed = time.perf_counter() - start
            result.update(
                training_time_sec=elapsed,
                train_time_sec=elapsed,
                optimization_time_sec=optimization,
                epochs_run=epoch,
                maximum_epochs=epochs,
                early_stopping_patience=settings["early_stopping_patience"],
                training_termination_reason=termination,
                best_epoch=best_epoch,
                validation_accuracy=best,
            )
            checkpoint = result_path.parent / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "validation_accuracy": best,
                },
                checkpoint,
            )
            result["checkpoint_path"] = str(checkpoint)
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                result["accuracy"] = (
                    (forward("test").argmax(-1) == data.y[data.test_mask]).float().mean().item()
                )

            class InferenceInterface(torch.nn.Module):
                def forward(self, ignored):
                    # All node predictions: mask is applied by the original node head.
                    return forward("all")

            data.all_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
            latency = request["config"]["training"]
            result.update(
                measure_latency(
                    InferenceInterface(),
                    None,
                    device,
                    latency["latency_warmup"],
                    latency["latency_repeats"],
                )
            )
            result.update(status="SUCCESS", failure_reason=None)
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (ImportError, ModuleNotFoundError)) or stage == "dependencies":
            status = "DEPENDENCY_ERROR"
        elif isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
            status = "OOM"
        elif stage == "preprocessing":
            status = "PREPROCESSING_ERROR"
        else:
            status = "RUNTIME_ERROR"
        result.update(status=status, failure_reason=message, error=message, failure_stage=stage)
    finally:
        if profiler is not None:
            result.update(profiler.metrics())
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    request_path, output_path = map(Path, sys.argv[1:])
    execute(json.loads(request_path.read_text(encoding="utf-8")), output_path)
