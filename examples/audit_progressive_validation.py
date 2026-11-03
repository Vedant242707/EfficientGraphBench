"""Replay native ogbn-arxiv finalists on official validation nodes, without training."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from efficientgraphbench.benchmark.interchange import canonical_array_hash, file_sha256
from efficientgraphbench.config import ModelConfig
from efficientgraphbench.models.registry import build_model
from efficientgraphbench.progressive.metrics import LABELS
from efficientgraphbench.training.evaluator import classification_metrics


def validation_ids(stage_ids, mask, official):
    actual = np.sort(stage_ids[mask])
    if not np.array_equal(actual, np.sort(official)):
        raise ValueError("Full-stage validation node IDs differ from OGB's official valid split")
    return len(actual)


def audit(saved, valid_file, models, device, *, expected_dataset="ogbn-arxiv"):
    payload = json.loads(saved.read_text(encoding="utf8"))
    meta = payload["metadata"]
    metric = meta["config"]["selection_metric"]
    if meta["dataset"] != expected_dataset or metric not in LABELS:
        raise ValueError(f"Expected a {expected_dataset} run with a recorded selection protocol")
    with gzip.open(valid_file, "rt") as stream:
        official = np.loadtxt(stream, dtype=np.int64, ndmin=1)
    rows = []
    graphs = {}
    for r in payload["records"]:
        f = r["fidelity"]
        if r["model"] not in models or r["status"] != "SUCCESS" or f["graph_fraction"] != 1:
            continue
        index = f["stage_index"]
        if index not in graphs:
            path = saved.parent / "graphs" / f"stage-{index + 1}" / "graph.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = dict(archive)
            if canonical_array_hash(arrays) != f["stage_graph_hash"]:
                raise ValueError("Stage graph or split differs from the recorded run")
            count = validation_ids(np.asarray(f["selected_node_ids"]), arrays["val_mask"], official)
            graphs[index] = (
                Data(**{k: torch.from_numpy(v) for k, v in arrays.items()}).to(device),
                count,
                f["stage_graph_hash"],
            )
        data, count, graph_hash = graphs[index]
        ckpt = saved.parent / "checkpoints" / r["model"] / str(r["seed"]) / f"{index}.pt"
        if file_sha256(ckpt) != r["checkpoint_id"]:
            raise ValueError("Checkpoint checksum mismatch")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if (state["stage_graph_hash"], state["model"], state["seed"]) != (
            graph_hash,
            r["model"],
            r["seed"],
        ):
            raise ValueError("Checkpoint identity mismatch")
        config = ModelConfig(**state["model_configuration"]["model"])
        model = build_model(config, data.num_node_features, f["num_classes"]).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            scores = classification_metrics(model(data), data.y, data.val_mask)
        delta = scores[metric.removeprefix("val_")] - r["validation_metric"]
        rows.append(
            dict(
                model=r["model"],
                seed=r["seed"],
                official_validation_nodes=count,
                validation_accuracy=scores["accuracy"],
                validation_macro_f1=scores["f1_macro"],
                recorded_selection=r["validation_metric"],
                delta=delta,
            )
        )
    if not rows:
        raise ValueError("No successful requested models at 100% graph size")
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print(
        "\nMean across replayed seeds:\n",
        frame.groupby("model")[
            ["validation_accuracy", "validation_macro_f1", "recorded_selection"]
        ].mean(),
    )
    if frame.delta.abs().max() > 1e-6:
        raise ValueError("Replay differs from the recorded scalar; do not treat it as identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", help="Saved progressive-results.json")
    parser.add_argument(
        "valid_file", type=Path, nargs="?", help="OGB ogbn_arxiv/split/time/valid.csv.gz"
    )
    parser.add_argument("--models", nargs="+", default=["gcn", "graphsage", "sgformer_pyg"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    roots = [root, root / "examples"]
    if args.results is None:
        pattern = (
            "experiments/college-reports-*/ogbn-arxiv/progressive-100pct/*/progressive-results.json"
        )
        files = [p for folder in roots for p in folder.glob(pattern)]
        if not files:
            parser.error("Pass the path to your ogbn-arxiv progressive-results.json.")
        args.results = max(files, key=lambda p: p.stat().st_mtime)
    if args.valid_file is None:
        files = [
            p
            for folder in roots
            for p in (folder / "data").glob("**/ogbn_arxiv/split/time/valid.csv.gz")
        ]
        if not files:
            parser.error("Pass the results JSON and the official OGB valid.csv.gz path.")
        args.valid_file = files[0]
    print("Results:", args.results, "\nOfficial validation split:", args.valid_file, flush=True)
    audit(args.results, args.valid_file, args.models, args.device)
