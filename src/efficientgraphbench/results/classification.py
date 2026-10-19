"""Conservative descriptive diagnostics, not significance tests."""

from collections import defaultdict
from math import ceil
from statistics import mean

from efficientgraphbench.benchmark.interchange import fingerprint
from efficientgraphbench.names import dataset_name, model_name

EXPLANATION = (
    "Precision, recall and F1 are weighted by test-class support. "
    "Macro gives each evaluated class equal weight; micro pools predictions and, "
    "for single-label classification, micro precision/recall/F1 equal accuracy. "
    "Attention notes require sufficient test support and repeated evidence; "
    "they are descriptive checks, not statistical significance tests."
)


def exceptional_notes(records):
    groups = defaultdict(list)
    for row in records:
        if (
            row.get("status") != "SUCCESS"
            or row.get("f1_macro") is None
            or row.get("f1_micro") is None
        ):
            continue
        cfg = row.get("config") or {}
        identity = {
            k: row.get(k)
            for k in (
                "dataset",
                "model",
                "graph_content_hash",
                "hardware_fingerprint",
                "source_snapshot",
            )
        }
        identity.update(
            model_config=cfg.get("model"),
            training=cfg.get("training"),
            options=cfg.get("model_options"),
        )
        groups[fingerprint(identity)].append(row)
    notes = []
    for rows in groups.values():
        rows = list({(r.get("seed"), r.get("split_hash")): r for r in rows}.values())
        if len(rows) < 5:
            continue
        supported = [
            r
            for r in rows
            if sum(c["support"] for c in r.get("classification_per_class", [])) >= 100
        ]
        parts = []
        if len(supported) >= 5:
            gaps = [abs(r["f1_macro"] - r["f1_micro"]) for r in supported]
            if mean(gaps) >= 0.15 and sum(g >= 0.15 for g in gaps) >= ceil(0.8 * len(gaps)):
                parts.append(
                    f"persistent macro/micro F1 gap: {mean(gaps) * 100:.1f} percentage points "
                    f"on average across {len(gaps)} runs with at least 100 test nodes"
                )
        classes = defaultdict(list)
        for row in rows:
            for c in row.get("classification_per_class", []):
                if c["support"] >= 20:
                    classes[c["class_id"]].append(c["recall"])
        for label, recalls in sorted(classes.items()):
            zeros = sum(v == 0 for v in recalls)
            if len(recalls) >= 5 and zeros >= ceil(0.8 * len(recalls)):
                parts.append(
                    f"class {label} has zero recall in {zeros}/{len(recalls)} runs "
                    "with at least 20 test examples of that class"
                )
        if parts:
            first = rows[0]
            notes.append(
                f"{dataset_name(str(first.get('dataset')))} / "
                f"{model_name(str(first.get('model')))}: " + "; ".join(parts) + "."
            )
    return notes
