"""Convert named CSV columns into the benchmark's explicit graph contract."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from efficientgraphbench.datasets.custom import load_custom
from efficientgraphbench.datasets.tables import read_table
from efficientgraphbench.datasets.validation import stratified_masks, validate_graph


def resolve_column(frame, explicit, aliases, option):
    if explicit is not None:
        if explicit not in frame:
            raise ValueError(f"No '{explicit}' column; specify {option}")
        return explicit
    matches = [name for name in frame if name.lower() in aliases]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot identify {option} uniquely; candidates: {matches}. "
            f"Specify {option} with a column name."
        )
    return matches[0]


def prepare_csv(
    nodes_path,
    edges_path,
    output_dir,
    *,
    node_id=None,
    label=None,
    source=None,
    target=None,
    features=None,
    split_column="split",
    split_seed=0,
    nodes_options=None,
    edges_options=None,
):
    nodes = read_table(nodes_path, **(nodes_options or {}))
    edges = read_table(edges_path, **(edges_options or {}))
    node_id = resolve_column(nodes, node_id, {"node_id", "id", "person_id"}, "--node-id")
    label = resolve_column(nodes, label, {"label", "category", "class", "target"}, "--label")
    source = resolve_column(edges, source, {"source", "from_id", "from", "src"}, "--source")
    target = resolve_column(edges, target, {"target", "to_id", "to", "dst"}, "--target")
    for name in (node_id, label):
        if name not in nodes:
            raise ValueError(f"nodes file has no '{name}' column; specify --node-id / --label")
    for name in (source, target):
        if name not in edges:
            raise ValueError(f"edges file has no '{name}' column; specify --source / --target")
    if node_id == label or source == target:
        raise ValueError("ID/label and source/target must refer to different columns")
    excluded = {node_id, label, split_column}
    if features is None:
        features = [
            column
            for column in nodes
            if column not in excluded
            and pd.to_numeric(nodes[column], errors="coerce").notna().all()
        ]
    if not features or len(features) != len(set(features)):
        raise ValueError("select at least one unique numeric feature column with --features")
    if any(column in excluded or column not in nodes for column in features):
        raise ValueError("features must exist and must not include node ID, label, or split")
    prepared = pd.DataFrame({"node_id": nodes[node_id], "label": nodes[label]})
    if split_column in nodes:
        prepared["split"] = nodes[split_column]
        split_policy = f"provided column: {split_column}"
    else:
        labels = sorted(set(nodes[label]) - {""})
        mapping = {value: index for index, value in enumerate(labels)}
        labeled = nodes[label] != ""
        values = torch.tensor(
            [mapping[value] for value in nodes.loc[labeled, label]], dtype=torch.long
        )
        masks = stratified_masks(values, split_seed)
        prepared["split"] = "unused"
        indices = nodes.index[labeled]
        for split_name, mask in zip(("train", "val", "test"), masks):
            prepared.loc[indices[mask.numpy()], "split"] = split_name
        split_policy = f"generated stratified 60/20/20; seed {split_seed}"
    for index, column in enumerate(features, 1):
        try:
            prepared[f"feature_{index}"] = pd.to_numeric(nodes[column], errors="raise")
        except ValueError as exc:
            raise ValueError(f"feature '{column}' must be numeric, with no missing values") from exc
    prepared_edges = edges[[source, target]].copy()
    prepared_edges.columns = ["source", "target"]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "output folder must be new or empty; existing datasets are not overwritten"
        )
    # Validate in a temporary sibling before publishing any output files.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary)
        prepared.to_csv(stage / "nodes.csv", index=False)
        prepared_edges.to_csv(stage / "edges.csv", index=False)
        data, classes, _ = load_custom(stage)
        validate_graph(data, classes)
        summary = {
            "columns": {
                "node_id": node_id,
                "label": label,
                "source": source,
                "target": target,
                "split": split_column if split_column in nodes else None,
            },
            "inputs": {
                "nodes": {"path": str(nodes_path), **(nodes_options or {})},
                "edges": {"path": str(edges_path), **(edges_options or {})},
            },
            "nodes": data.num_nodes,
            "edges": data.num_edges,
            "classes": classes,
            "features": features,
            "split": split_policy,
            "ignored_columns": [c for c in nodes if c not in excluded and c not in features],
        }
        (stage / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        nodes.to_csv(stage / "original_nodes.csv", index=False)
        edges.to_csv(stage / "original_edges.csv", index=False)
        output.mkdir(exist_ok=True)
        for name in (
            "nodes.csv",
            "edges.csv",
            "preparation.json",
            "original_nodes.csv",
            "original_edges.csv",
        ):
            (stage / name).replace(output / name)
    return summary
