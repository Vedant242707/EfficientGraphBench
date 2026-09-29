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
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
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
    return paths
