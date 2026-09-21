"""Generate current provenance without claiming unexecuted reference measurements."""

import hashlib
import inspect
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from efficientgraphbench.config import ModelConfig
from efficientgraphbench.models.catalog import ROOT, model_spec
from efficientgraphbench.models.registry import build_model
from efficientgraphbench.names import MODEL_NAMES
from efficientgraphbench.paths import WORKER


def generate_provenance(output=None):
    previous_path = ROOT / "audits/review20/model_provenance.json"
    old = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else []
    previous = {r.get("model_id"): r for r in old}
    alias = {
        "graphgps_adapted": "graphgps",
        "graphormer_adapted": "graphormer",
        "sgformer_pyg": "sgformer",
        "appnp_adapted": "appnp",
    }
    measurements = []
    for path in (ROOT / "experiments").glob("reference-*/raw/results.jsonl"):
        measurements.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    rows = []
    for name, display in MODEL_NAMES.items():
        spec = model_spec(name)
        row = dict(previous.get(name, previous.get(alias.get(name), {})))
        row.update(
            schema_version=3,
            model=display,
            model_id=name,
            source_type=spec["source_type"],
            category=spec["category"],
            environment=spec["environment"],
            repository_url=spec.get("repository_url"),
            repository_commit=spec.get("commit"),
        )
        if spec["runner"] == "in_process":
            config = ModelConfig(name=name)
            model = build_model(config, 1433, 7)
            underlying = model.model
            row.update(
                implementation_class=f"{type(underlying).__module__}.{type(underlying).__name__}",
                wrapper_class="efficientgraphbench.models.registry.GraphModel",
                implementation_source=inspect.getfile(type(underlying)),
                underlying_layer_classes=sorted(
                    {f"{type(m).__module__}.{type(m).__name__}" for m in model.modules()}
                ),
                parameter_count=sum(p.numel() for p in model.parameters()),
                trainable_parameter_count=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                parameter_count_context={
                    "dataset": "Cora",
                    "in_channels": 1433,
                    "out_channels": 7,
                    "model_config": asdict(config),
                },
                dependencies={
                    "torch": version("torch"),
                    "torch-geometric": version("torch-geometric"),
                },
                package_name="torch-geometric"
                if spec["source_type"] == "library"
                else "efficientgraphbench",
                package_version=version("torch-geometric")
                if spec["source_type"] == "library"
                else version("efficientgraphbench"),
            )
            row["architecture_modified"] = spec["source_type"] not in {"library", "official_repo"}
            row["architecture_modifications"] = row.get("exact_modifications", [])
            row["task_modifications"] = ["Full-graph node classification with canonical masks"]
            row["preprocessing_requirements"] = (
                ["Dense shortest paths; custom 4096-node safeguard"]
                if name == "graphormer_adapted"
                else ["Canonical graph tensors"]
            )
            source = Path(inspect.getfile(type(underlying)))
            wrapper = Path(inspect.getfile(type(model)))
            row.update(
                source_file=str(source),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                wrapper_source={
                    "file": str(wrapper),
                    "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                },
            )
            row["layer_source_files"] = []
            for cls in {type(layer) for layer in model.modules()}:
                path = Path(inspect.getfile(cls))
                row["layer_source_files"].append(
                    {
                        "class": f"{cls.__module__}.{cls.__name__}",
                        "file": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            if name == "appnp":
                row.update(
                    package_name=None,
                    package_version=None,
                    reference_commit_version=spec["commit"],
                    implementation_modified=False,
                    architecture_modified=False,
                    architecture_modifications=[],
                    exact_modifications=[
                        "Execute the unchanged upstream Net class without "
                        "top-level CLI/training side effects; provide args "
                        "and dataset dimensions through the adapter"
                    ],
                    verification={
                        "method": "Pinned source hash and runtime layer inspection",
                        "architecture": "input dropout, Linear/ReLU, dropout, "
                        "Linear, PyG APPNP, log_softmax",
                        "propagation_class": "torch_geometric.nn.conv.appnp.APPNP",
                    },
                )
        else:
            # Do not carry old custom/PyG fields into the new reference identity.
            paper = row.get("original_paper")
            row = {
                k: row[k]
                for k in (
                    "schema_version",
                    "model",
                    "model_id",
                    "source_type",
                    "category",
                    "environment",
                    "repository_url",
                    "repository_commit",
                )
            }
            row.update(
                original_paper=paper or spec.get("paper"),
                original_repository_url=spec.get("repository_url"),
                implementation_source=spec.get("repository_url"),
                package_name=None,
                package_version=None,
                reference_commit_version=spec.get("commit"),
                implementation_class=None,
                wrapper_class=None,
                adapter_entry_point="runners.reference_worker.execute",
                implementation_modified=False,
                architecture_modified=False,
                exact_modifications=[],
                architecture_modifications=[],
                task_modifications=[],
                underlying_layer_classes=[],
                parameter_count=None,
                trainable_parameter_count=None,
                preprocessing_requirements=[],
                dependencies={},
                status="UNSUPPORTED_TASK" if spec["runner"] == "unsupported" else "UNMEASURED",
            )
            worker_path = WORKER
            row["wrapper_source"] = {
                "file": str(worker_path),
                "sha256": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
            }
            candidate_id = "sgformer" if name == "sgformer_reference" else name
            candidates = [
                r
                for r in measurements
                if r.get("model_id") in {name, candidate_id}
                and r.get("status") == "SUCCESS"
                and r.get("repository_commit") == spec.get("commit")
            ]
            if candidates:
                cora_candidates = [r for r in candidates if r.get("dataset") == "cora"]
                candidates = cora_candidates or candidates
                latest = max(candidates, key=lambda r: (r.get("epochs_run", 0), r["run_id"]))
                for key in (
                    "implementation_class",
                    "underlying_layer_classes",
                    "parameter_count",
                    "trainable_parameter_count",
                    "dependencies",
                    "architecture_modified",
                    "architecture_modifications",
                    "task_modifications",
                    "preprocessing_requirements",
                ):
                    row[key] = latest.get(key)
                row.update(
                    status="VERIFIED_RUNTIME",
                    measured_run_id=latest["run_id"],
                    parameter_count_context={
                        "dataset": latest["dataset"],
                        "recipe": latest.get("recipe"),
                    },
                    exact_modifications=latest.get("task_modifications", []),
                )
            if spec["runner"] == "unsupported":
                row["reason"] = spec["reason"]
        from efficientgraphbench.models.metadata import get_model

        metadata = get_model(name)
        row.setdefault("original_paper", metadata.get("paper"))
        row.setdefault("original_repository_url", spec.get("repository_url"))
        row.setdefault("reference_commit_version", spec.get("commit") or row.get("package_version"))
        row.setdefault("implementation_modified", row.get("architecture_modified", False))
        row.setdefault("exact_modifications", row.get("architecture_modifications", []))
        rows.append(row)
    output = Path(output) if output else previous_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
