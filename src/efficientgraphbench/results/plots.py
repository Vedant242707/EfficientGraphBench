"""Publication exports from a single experiment's summary; no training."""

from pathlib import Path

import numpy as np
import pandas as pd

from efficientgraphbench.names import dataset_name, model_name


def plot_summary(summary, output_dir):
    """Export accuracy/latency and GPU memory plots as PNG/PDF.

    Accept a summary DataFrame or CSV path from one experiment/dataset.
    GPU bars are means of per-run peaks, not instantaneous memory.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ValueError("Install plotting support: python -m pip install matplotlib") from exc

    frame = summary.copy() if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    required = {
        "dataset",
        "model",
        "runs",
        "accuracy_mean",
        "accuracy_sd",
        "inference_latency_median_ms",
        "peak_gpu_allocated_mib",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing summary columns: {', '.join(sorted(missing))}")
    if frame["dataset"].nunique() != 1:
        raise ValueError("Plot one dataset/experiment at a time")
    successful = frame[pd.to_numeric(frame["runs"], errors="coerce") > 0].copy()
    if successful.empty:
        raise ValueError("No successful measurements to plot")
    if successful["model"].duplicated().any():
        raise ValueError("Multiple groups for one model: select a single experiment/configuration")
    for column in required - {"dataset", "model"}:
        successful[column] = pd.to_numeric(successful[column], errors="coerce")
    title = dataset_name(str(successful.iloc[0]["dataset"]))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#555555",
        "#7B3294",
        "#8C510A",
        "#1B7837",
    ]
    notes = f"Successful runs/model: {int(successful.runs.min())}–{int(successful.runs.max())}. "
    failed = int(len(frame) - len(successful))
    notes += f"Models without successful runs excluded: {failed}."
    paths = []

    def save(fig, name):
        FigureCanvasAgg(fig)
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            _dark_figure(fig)
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
            paths.append(path)

    scatter = successful[
        np.isfinite(successful.accuracy_mean)
        & np.isfinite(successful.inference_latency_median_ms)
        & (successful.inference_latency_median_ms > 0)
    ]
    if scatter.empty:
        raise ValueError("No valid accuracy/latency pairs")
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    for i, (_, row) in enumerate(scatter.iterrows()):
        sd = row.accuracy_sd
        ax.errorbar(
            row.inference_latency_median_ms,
            row.accuracy_mean * 100,
            yerr=sd * 100 if pd.notna(sd) and sd >= 0 and row.runs > 1 else None,
            fmt="o",
            markersize=8,
            capsize=4,
            color=colors[i % len(colors)],
            label=f"{model_name(row.model)} (n={int(row.runs)})",
        )
    ax.set(
        xscale="log",
        xlabel="Inference latency (ms, log scale; lower is better)",
        ylabel="Test accuracy (%, higher is better)",
        title=f"{title} · Accuracy vs inference latency",
    )
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(
        0.08,
        -0.02,
        "Points: mean accuracy and mean per-run median latency. Bars: ±1 accuracy SD.\n" + notes,
        fontsize=9,
        color="#444444",
    )
    save(fig, "accuracy_vs_latency")

    memory = successful[
        np.isfinite(successful.peak_gpu_allocated_mib) & (successful.peak_gpu_allocated_mib >= 0)
    ]
    if not memory.empty:
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        values = memory.peak_gpu_allocated_mib.to_numpy()
        bars = ax.bar([model_name(m) for m in memory.model], values, color="#0072B2", width=0.65)
        ax.bar_label(bars, labels=[f"{v:,.1f}" for v in values], padding=4, fontsize=9)
        ax.set(
            ylabel="Mean peak GPU allocated memory (MiB)", title=f"{title} · GPU memory allocation"
        )
        ax.set_ylim(0, max(float(values.max()) * 1.18, 1))
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.text(
            0.08,
            -0.06,
            "Mean of per-run PyTorch allocated-memory peaks; excludes reserved-only memory.\n"
            f"Missing GPU measurements omitted: {len(successful) - len(memory)}. " + notes,
            fontsize=9,
            color="#444444",
        )
        save(fig, "gpu_memory_allocation")
    successful.to_csv(output / "plot_data.csv", index=False)
    if {"num_nodes", "training_time_sec"} <= set(successful.columns):
        paths.extend(plot_training_time(successful, output))
    return paths


def plot_training_time(records, output_dir):
    """Nodes vs mean training seconds, with one line per dataset/model/protocol.

    Accept per-run records, BenchmarkResults, DataFrame, or JSON/JSONL/CSV.
    Failed/skipped runs are excluded. Raw repeated runs get sample SD bars.
    Separate hardware and training recipes are never joined into one line.
    """
    import json

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ValueError("Install plotting support: python -m pip install matplotlib") from exc

    from efficientgraphbench.benchmark.interchange import fingerprint

    if isinstance(records, (str, Path)):
        path = Path(records)
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        elif path.suffix == ".json":
            records = json.loads(path.read_text())
        else:
            records = pd.read_csv(path)
    if hasattr(records, "records"):
        records = records.records
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")
    if not isinstance(records, list):
        raise ValueError("Expected run records, not scaling-summary.json; use scaling-results.json")
    points = []
    for row in records:
        if "runs" in row:
            if not row["runs"] or row["runs"] <= 0:
                continue
        elif row.get("status") != "SUCCESS":
            continue
        nodes, seconds = row.get("num_nodes"), row.get("training_time_sec")
        if nodes is None or seconds is None:
            continue
        nodes, seconds = float(nodes), float(seconds)
        if not np.isfinite(nodes) or not np.isfinite(seconds) or nodes <= 0 or seconds < 0:
            continue
        config = row.get("config") or {}
        context = row.get("scaling_context") or {}
        # CSV exports serialize nested records as JSON strings.
        if isinstance(config, str):
            config = json.loads(config)
        if isinstance(context, str):
            context = json.loads(context)
        dataset = context.get("parent_dataset", row.get("dataset", "unknown"))
        protocol = fingerprint(
            {
                "hardware": row.get("hardware_fingerprint"),
                "training": config.get("training"),
                "model": config.get("model"),
                "options": config.get("model_options"),
                "software": row.get("software"),
                "source": row.get("repository_commit"),
                "parent": context.get("parent_fingerprint"),
                "summary_group": row.get("comparison_group") if "runs" in row else None,
            }
        )[:8]
        points.append(
            dict(
                dataset=dataset,
                model=row.get("model", "unknown"),
                protocol=protocol,
                num_nodes=int(nodes),
                training_time_sec=seconds,
            )
        )
    if not points:
        raise ValueError("No successful node-count/training-time measurements to plot")
    frame = pd.DataFrame(points)
    grouped = frame.groupby(["dataset", "model", "protocol", "num_nodes"], as_index=False).agg(
        training_time_mean_sec=("training_time_sec", "mean"),
        training_time_sd_sec=("training_time_sec", "std"),
        measurements=("training_time_sec", "size"),
    )
    grouped["training_time_sd_sec"] = grouped.training_time_sd_sec.fillna(0)
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for (dataset, model, protocol), rows in grouped.groupby(["dataset", "model", "protocol"]):
        rows = rows.sort_values("num_nodes")
        ax.errorbar(
            rows.num_nodes,
            rows.training_time_mean_sec,
            yerr=rows.training_time_sd_sec,
            marker="o",
            capsize=3,
            label=f"{dataset_name(dataset)} / {model_name(model)} [{protocol}]",
        )
    ax.set(
        xlabel="Number of nodes",
        ylabel="Training time (seconds)",
        title="Number of nodes vs training time",
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(
        0.08,
        -0.03,
        "Successful measurements only; bars show sample SD across run records.\n"
        "One size gives one point. Scaling trains a fresh model at each size; "
        "epochs/early stopping affect time.",
        fontsize=9,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output / "nodes_vs_training_time.csv", index=False)
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"nodes_vs_training_time.{extension}"
        _dark_figure(fig)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    return paths


def _dark_figure(fig):
    from matplotlib.text import Text

    fig.patch.set_facecolor("#0B1020")
    for ax in fig.axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#B8C5D9")
        for spine in ax.spines.values():
            spine.set_color("#293548")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#172033")
            legend.get_frame().set_edgecolor("#293548")
    for item in fig.findobj(Text):
        item.set_color("#E8EEF7")
