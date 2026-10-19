"""Readable presentation; numeric exports remain unchanged."""

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from efficientgraphbench.names import dataset_name, model_name
from efficientgraphbench.results.classification import EXPLANATION, exceptional_notes
from efficientgraphbench.results.presentation import failure_summaries, successful_records


def number(value, percent=False):
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:,.3f}"


def show_results(results, console=None):
    console = console or Console()
    from efficientgraphbench.api.results import BenchmarkResults

    original = results.records
    visible = BenchmarkResults(successful_records(original))
    visible.aggregate_splits = results.aggregate_splits
    frame = visible.summary()
    if not original:
        console.print("No benchmark results yet.")
        return
    for dataset, rows in (
        frame.groupby("dataset", dropna=False, sort=False) if not frame.empty else []
    ):
        console.print(Text(f"{dataset_name(str(dataset))} • Benchmark results", style="bold cyan"))
        quality = Table(
            title="Classification • test set",
            box=box.SIMPLE_HEAVY,
            header_style="bold #163b50 on #dcebf2",
            padding=(0, 1),
        )
        for heading in (
            "Model",
            "Accuracy ± SD",
            "Precision",
            "Recall",
            "F1",
            "Macro-F1",
            "Micro-F1",
        ):
            quality.add_column(heading, justify="left" if heading == "Model" else "right")
        performance = Table(
            title="Performance • successful-run averages",
            box=box.SIMPLE_HEAVY,
            header_style="bold #163b50 on #dcebf2",
            padding=(0, 1),
        )
        for heading in (
            "Model",
            "Runs OK / total",
            "Train s",
            "Latency ms",
            "GPU alloc MiB",
            "GPU reserved MiB",
            "CPU MiB",
        ):
            performance.add_column(heading, justify="left" if heading == "Model" else "right")
        duplicated = rows.model.duplicated(keep=False)
        for index, (_, row) in enumerate(rows.iterrows(), 1):
            name = model_name(str(row.model))
            if duplicated.iloc[index - 1]:
                name += f" [{index}]"
                console.print(
                    Text(f"[{index}] {name}: comparison {row.comparison_group}", style="dim")
                )
            flagged = bool(row.classification_alert)
            if flagged:
                name += " *"
            accuracy = number(row.accuracy_mean, True)
            if pd.notna(row.accuracy_sd):
                accuracy += f" ± {row.accuracy_sd * 100:.2f}"
            quality.add_row(
                Text(name),
                accuracy,
                *[
                    Text(
                        number(row[k], True),
                        style="bold yellow" if flagged and k.startswith("f1_") else "",
                    )
                    for k in ("precision", "recall", "f1", "f1_macro", "f1_micro")
                ],
            )
            performance.add_row(
                Text(name),
                f"{row.runs} / {row.attempted_runs}",
                *[
                    number(row[k])
                    for k in (
                        "training_time_sec",
                        "inference_latency_median_ms",
                        "peak_gpu_allocated_mib",
                        "peak_gpu_reserved_mib",
                        "peak_cpu_memory_mb",
                    )
                ],
            )
        console.print(quality)
        console.print(
            "Scores are percentages; accuracy SD is in percentage points. "
            "Precision / Recall / F1 are support-weighted.",
            style="dim",
        )
        console.print(performance)
        console.print(
            "— = unavailable. Runs OK / total includes failed or skipped attempts.", style="dim"
        )
    failures = failure_summaries(original)
    if failures:
        console.print("Models not completed", style="bold yellow")
        for row in failures:
            console.print(
                Text(
                    f"• {row['Model']}: {row['Status']} — {row['Reason']} "
                    f"({row['Skipped']} skipped). {row['Note']}"
                )
            )
    notes = exceptional_notes(visible.records)
    if notes:
        console.print("* Patterns to review", style="bold yellow")
        for note in notes:
            console.print(Text("• " + note))
    console.print(Text("Note: " + EXPLANATION))
