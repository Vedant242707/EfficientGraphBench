from pathlib import Path

import pandas as pd

from efficientgraphbench.datasets.catalog import canonical_dataset
from efficientgraphbench.names import dataset_name, model_name
from efficientgraphbench.results.status import failure_reason, is_success, status_of
from efficientgraphbench.results.writer import latest_results, read_results


def write_report(dataset: str, output_dir: str) -> Path:
    dataset = canonical_dataset(dataset)
    records = latest_results(read_results(output_dir), dataset)
    successful = [row for row in records if is_success(row)]
    lines = [
        f"# {dataset_name(dataset)} benchmark report",
        "",
        f"{len(successful)} successful runs; {len(records) - len(successful)} failed/OOM.",
        "",
        "Accuracy is test accuracy at the best validation checkpoint. "
        "Runs use full-graph, transductive node classification. "
        "The dataset split and preprocessing are listed for each comparison group.",
        "",
    ]
    if successful:
        for group_id, group in pd.DataFrame(successful).groupby("comparison_group"):
            first = group.iloc[0]
            lines.extend(
                [
                    f"## Comparison {group_id}",
                    "",
                    f"Device: {first['device']} / {first['device_name']}. "
                    "Software and recipes may differ; see each run's provenance.",
                    "",
                    f"Split: {first.get('split', 'not recorded')}. "
                    f"Features: {first.get('feature_preprocessing', 'not recorded')}. "
                    f"Edges: {first.get('edge_policy', 'not recorded')}.",
                    "",
                    f"Dataset fingerprint: {first.get('dataset_fingerprint', 'not recorded')}.",
                    "",
                    f"Seeds: {', '.join(str(s) for s in sorted(group.seed.unique()))}.",
                    "",
                    "| Model | Source | Runs | Accuracy mean ± SD | Preprocess s | Train s | "
                    "Latency median ms | GPU allocated MiB | GPU reserved MiB | "
                    "Sampled CPU peak MiB |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for model, rows in group.groupby("model"):
                sd = rows.accuracy.std(ddof=1) if len(rows) > 1 else 0
                gpu = rows.peak_gpu_memory_mb.mean()
                gpu_text = f"{gpu:.1f}" if pd.notna(gpu) else "N/A"
                prep = (
                    rows.preprocessing_time_sec.mean() if "preprocessing_time_sec" in rows else None
                )
                reserved = (
                    rows.peak_gpu_reserved_mib.mean() if "peak_gpu_reserved_mib" in rows else None
                )
                lines.append(
                    f"| {model_name(model)} | {rows.iloc[0].get('source_type', 'legacy')} | "
                    f"{len(rows)} | "
                    f"{rows.accuracy.mean():.4f} ± {sd:.4f} | "
                    f"{prep if prep is not None else 'N/A'} | "
                    f"{rows.train_time_sec.mean():.3f} | {rows.inference_ms.mean():.3f} | "
                    f"{gpu_text} | {reserved if reserved is not None else 'N/A'} | "
                    f"{rows.peak_cpu_memory_mb.mean():.1f} |"
                )
            lines.append("")
            lines.extend(["### Per-run recipes and budgets", ""])
            for row in group.to_dict("records"):
                lines.extend(
                    [
                        f"- {model_name(row['model'])}, seed {row['seed']}: "
                        f"{row.get('epochs_run', 'unknown')} epochs; "
                        f"optimizer {row.get('optimizer', 'unknown')}; "
                        f"LR {row.get('learning_rate', 'unknown')}; "
                        f"scheduler {row.get('scheduler')}; "
                        f"termination {row.get('training_termination_reason', 'unknown')}. "
                        f"Environment: {row.get('environment', 'legacy')}; "
                        f"source commit: {row.get('repository_commit')}. "
                        f"Exact config and provenance: raw/{row['run_id']}.json",
                    ]
                )
            lines.append("")
    failures = [row for row in records if not is_success(row)]
    if failures:
        lines.extend(
            [
                "## Failed runs",
                "",
                "| Model | Status | Nodes | Edges | Reason | Limit |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for row in failures:
            values = [
                model_name(row["model"]),
                status_of(row),
                row.get("num_nodes"),
                row.get("num_edges"),
                failure_reason(row),
                row.get("implementation_limit"),
            ]
            lines.append(
                "| "
                + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in values)
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "These measure predictive accuracy and practical implementation cost on this hardware, "
            "not pure architecture-only efficiency or paper score reproductions. "
            "Model-specific recipes and software versions can differ. "
            "Training time includes validation and best-checkpoint CPU copies. "
            "Latency is the mean across runs of each run's median warmed full-graph inference. "
            "Memory values are means of per-run peaks. CPU RSS includes process/library memory "
            "and allocator retention from earlier runs in the sequential matrix. It is not "
            "isolated model memory. Reference workers have separate processes. "
            "GPU allocated and reserved are distinct PyTorch allocator metrics, not NVML VRAM. "
            "Preprocessing is explicit; reference peak memory includes model preprocessing, "
            "initialization, training and inference. "
            "Old schema-3 in-process peaks excluded preprocessing. "
            "SD is sample standard deviation across saved runs; repeated seeds are not "
            "independent trials. Results apply to this dataset and machine only.",
            "",
            "Raw records, per-run configurations, logs, and checkpoints are stored alongside "
            "this report in the experiment directory.",
            "",
        ]
    )
    path = Path(output_dir) / "reports" / f"{dataset}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
