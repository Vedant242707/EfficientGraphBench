"""One-command analysis: readable metrics, selected candidates and embedded plots."""

import base64
import html
import json
from pathlib import Path

import pandas as pd

from efficientgraphbench.names import dataset_name, model_name
from efficientgraphbench.results.classification import EXPLANATION, exceptional_notes
from efficientgraphbench.results.pareto import pareto_frontier
from efficientgraphbench.results.presentation import failure_summaries, successful_records


def load_analysis_results(path):
    from efficientgraphbench.api.results import BenchmarkResults
    from efficientgraphbench.results.writer import read_results

    path = Path(path)
    scaling = path / "scaling-results.json" if path.is_dir() else path
    if scaling.name == "scaling-results.json" and scaling.exists():
        all_rows = json.loads(scaling.read_text(encoding="utf-8"))
        latest = {}
        for row in all_rows:
            if row.get("status") == "SUCCESS":
                latest[row["model"]] = row["num_nodes"]
        rows = [
            r
            for r in all_rows
            if r.get("num_nodes") == latest.get(r.get("model")) or r.get("model") not in latest
        ]
        result = BenchmarkResults(rows, scaling.parent)
        result.all_results_path = scaling
    else:
        if path.is_dir():
            rows = read_results(path)
        elif path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Expected run records or experiment directory")
        result = BenchmarkResults(rows, path if path.is_dir() else path.parent)
    result.aggregate_splits = True
    return result


def _table(frame):
    frame = frame.astype(object).where(pd.notna(frame), "—")
    return frame.to_html(
        index=False, escape=True, border=0, na_rep="—", float_format=lambda v: f"{v:.3f}"
    )


def comparison_criteria(group):
    """Explain only the objectives actually used to select the frontier."""
    quality = (
        (
            "Macro-F1 — higher is better",
            "Balances precision and recall while giving each class equal weight. "
            "Useful when performance on smaller classes matters as much as larger classes.",
        )
        if group["quality_metric"] == "f1_macro"
        else (
            "Accuracy — higher is better",
            "The proportion of test nodes classified correctly. "
            "Useful when overall prediction correctness is the priority; "
            "common classes influence it more.",
        )
    )
    criteria = [
        quality,
        (
            "Inference latency — lower is better",
            "Time needed to produce predictions. Lower measured latency suits faster responses.",
        ),
    ]
    if group["memory_metric"] == "peak_gpu_reserved_mib":
        criteria.append(
            (
                "Peak GPU reserved memory — lower is better",
                "The largest PyTorch allocator reservation across runs. Lower values indicate less "
                "reserved GPU memory on this test, useful when GPU capacity is limited; "
                "this is not total device memory usage.",
            )
        )
    elif group["memory_metric"] == "peak_cpu_memory_mb":
        criteria.append(
            (
                "Peak process RAM — lower is better",
                "The largest measured process memory peak across runs. Useful for comparing RAM "
                "requirements, but includes libraries and retained allocations, "
                "not just the model.",
            )
        )
    if "training_time_sec" in group["objectives"]:
        criteria.append(
            (
                "Training time — lower is better",
                "Measured time spent training. Lower values suit "
                "shorter training or retraining cycles.",
            )
        )
    return criteria


def group_setup_notes(group, groups):
    settings = group.get("comparison_settings", {})
    epochs = settings.get("maximum_epochs")
    patience = settings.get("early_stopping_patience")
    notes = [
        f"Training limit: {epochs} epochs per run."
        if epochs is not None
        else "Training limit: not recorded."
    ]
    notes.append(
        "Early stopping: disabled. Training can use the full epoch limit."
        if patience is None
        else f"Early stopping: stop after {patience} epochs without validation improvement."
    )
    labels = {
        "dataset": "dataset",
        "graph_content_hash": "graph contents",
        "hardware_fingerprint": "hardware",
        "device": "device",
        "num_nodes": "node count",
        "num_edges": "edge count",
        "profiler_settings": "measurement settings",
        "source_snapshot": "benchmark code version",
        "feature_preprocessing": "feature preprocessing",
        "edge_policy": "edge preprocessing",
        "batch_size": "batch size",
        "maximum_epochs": "maximum epochs",
        "early_stopping_patience": "early-stopping rule",
        "split_mode": "split method",
    }
    others = [g for g in groups if g is not group and g["dataset"] == group["dataset"]]
    differences = [
        label
        for key, label in labels.items()
        if any(settings.get(key) != g.get("comparison_settings", {}).get(key) for g in others)
    ]
    if differences:
        notes.append(
            "Why this group is separate: different "
            + ", ".join(differences)
            + ". The report compares matching test settings together."
        )
    if epochs is not None and patience is not None and patience >= epochs:
        notes.append(
            "Here the waiting period is as long as the entire training limit, so this "
            "early-stopping rule cannot meaningfully shorten training. The current grouping "
            "rule still treats it as different from disabled early stopping."
        )
    if sum(c["qualified"] for c in group["candidates"]) < 2:
        notes.append("No competing eligible models in this group; best-model labels are not shown.")
    return notes


def show_all(results, output_dir=None, *, display=True, open_browser=False, **options):
    """Export a self-contained HTML analysis, display inline in Jupyter, return paths.

    Never trains or downloads datasets. Missing plots carry explicit reasons.
    Scaling uses all saved sizes for the curve and last-size records for tables.
    """
    from efficientgraphbench.api.results import BenchmarkResults
    from efficientgraphbench.results.plots import plot_summary, plot_training_time

    output = Path(output_dir or (results.output_dir or Path("experiments")) / "analysis")
    output.mkdir(parents=True, exist_ok=True)
    curve_records = results.records
    if getattr(results, "all_results_path", None):
        curve_records = json.loads(Path(results.all_results_path).read_text(encoding="utf-8"))
        results = load_analysis_results(results.all_results_path)
    original_records = results.records
    failures = failure_summaries(original_records)
    visible = BenchmarkResults(successful_records(original_records), results.output_dir)
    visible.aggregate_splits = results.aggregate_splits
    results = visible
    analysis = pareto_frontier(original_records, **options)
    (output / "pareto.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8"
    )
    summary = results.summary(detailed=True)
    summary.to_csv(output / "summary.csv", index=False)
    sections = [
        "<h1>Static Full Benchmark — EfficientGraphBench results</h1>",
        "<p>Measured test performance and resource use. "
        "No training was performed by this report.</p>",
    ]
    chart_paths, notices = [], []

    def add_plot(path, caption):
        if path.suffix != ".png":
            return
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        sections.append(
            f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{payload}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    def format_scores(frame):
        frame = frame.copy()
        for key in frame:
            if key in {"accuracy_mean", "precision", "recall", "f1", "f1_macro", "f1_micro"}:
                frame[key] = frame[key].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
        return frame

    if not summary.empty:
        for dataset, rows in summary.groupby("dataset", sort=False):
            rows = rows.copy()
            rows["model"] = rows.model.map(model_name)
            duplicate_models = rows.model.duplicated(keep=False)
            for i in rows.index[duplicate_models]:
                rows.loc[i, "model"] += f" (configuration {list(rows.index).index(i) + 1})"
            rows["accuracy_sd_pp"] = rows.accuracy_sd * 100
            rows["accuracy_sd_pp"] = rows.accuracy_sd_pp.map(
                lambda value: f"{value:.2f}" if pd.notna(value) else "—"
            )
            rows["num_nodes"] = rows.num_nodes.map(
                lambda value: f"{value:,.0f}" if pd.notna(value) else "—"
            )
            rows["Alert"] = rows.classification_alert.map({True: "* Review", False: ""})
            sections.append(
                f"<h2>{html.escape(dataset_name(str(dataset)))}</h2><h3>Classification</h3>"
            )
            columns = [
                "model",
                "runs",
                "accuracy_mean",
                "accuracy_sd_pp",
                "precision",
                "recall",
                "f1",
                "f1_macro",
                "f1_micro",
                "Alert",
            ]
            labels = {
                "model": "Model",
                "runs": "Successful runs",
                "accuracy_mean": "Accuracy",
                "accuracy_sd_pp": "Accuracy SD (pp)",
                "precision": "Precision (weighted)",
                "recall": "Recall (weighted)",
                "f1": "F1 (weighted)",
                "f1_macro": "Macro-F1",
                "f1_micro": "Micro-F1",
            }
            sections.append(_table(format_scores(rows[columns]).rename(columns=labels)))
            sections.append("<h3>Performance</h3>")
            columns = [
                "model",
                "num_nodes",
                "runs",
                "attempted_runs",
                "training_time_sec",
                "inference_latency_median_ms",
                "peak_gpu_allocated_mib",
                "peak_gpu_reserved_mib",
                "peak_cpu_memory_mb",
                "status",
            ]
            sections.append(
                _table(
                    rows[columns].rename(
                        columns={
                            "model": "Model",
                            "num_nodes": "Nodes",
                            "runs": "OK",
                            "attempted_runs": "Total",
                            "training_time_sec": "Train (s)",
                            "inference_latency_median_ms": "Latency (ms)",
                            "peak_gpu_allocated_mib": "GPU allocated (MiB)",
                            "peak_gpu_reserved_mib": "GPU reserved (MiB)",
                            "peak_cpu_memory_mb": "CPU RSS (MiB)",
                        }
                    )
                )
            )
        for note in exceptional_notes(results.records):
            sections.append(f'<p class="alert">* {html.escape(note)}</p>')
    else:
        sections.append("<p>No complete model results available.</p>")
    sections.append(f'<p class="note">{html.escape(EXPLANATION)}</p>')
    if failures:
        sections.append("<h3>Models not completed</h3>")
        sections.append(_table(pd.DataFrame(failures)))
    sections.append(
        "<h2>Pareto Comparison</h2>"
        "<p>Statistics for all eligible models, with frontier membership shown separately. "
        "Selection uses the configured quality/resource objectives and limits. "
        "No overall winner is assigned.</p>"
    )
    lookup = {r.get("run_id"): r for r in results.records}
    for group in analysis["groups"]:
        device_label = "GPU (CUDA)" if str(group["device"]).startswith("cuda") else "CPU"
        sections.append(
            f"<h3>{html.escape(dataset_name(str(group['dataset'])))} · "
            f"{group['num_nodes']:,} nodes · {device_label}</h3>"
        )
        sections.append('<section class="criteria"><h4>Training setup and grouping</h4>')
        for note in group_setup_notes(group, analysis["groups"]):
            sections.append(f"<p>{html.escape(note)}</p>")
        sections.append("</section>")
        sections.append('<section class="criteria"><h4>Selection criteria</h4><ul>')
        for title, explanation in comparison_criteria(group):
            sections.append(
                f"<li><strong>{html.escape(title)}</strong><br>{html.escape(explanation)}</li>"
            )
        sections.append(
            "</ul><p>These factors are considered together. A model is on the Pareto frontier "
            "when no other eligible model is at least as good on every selected factor and "
            "better on at least one. This identifies candidates worth considering for your "
            "accuracy, speed and resource priorities; it does not establish a universal winner.</p>"
            "</section>"
        )
        candidates = group["candidates"]
        if group["memory_metric"] == "peak_cpu_memory_mb":
            sections.append(
                "<p>CPU RSS includes process/library memory and allocator retention; "
                "it is not isolated model memory.</p>"
            )
        if len(candidates) == 1:
            sections.append(
                "<p>Only one complete candidate is available; this does not "
                "establish superiority over other models.</p>"
            )
        if candidates:
            table = pd.DataFrame(candidates)[
                [
                    "model",
                    "candidate",
                    "runs",
                    "quality",
                    "quality_sd",
                    "latency_ms",
                    "memory_mib",
                    "training_time_sec",
                    "qualified",
                    "frontier",
                ]
            ]
            table["candidate"] = [f"Configuration {i + 1}" for i in range(len(table))]
            table["model"] = table.model.map(model_name)
            for column in ("quality", "quality_sd"):
                table[column] = table[column].map(lambda value: f"{value:.2%}")
            if not table.model.duplicated().any():
                table = table.drop(columns="candidate")
            candidate_table = _table(
                table.rename(
                    columns={
                        "quality": "Macro-F1"
                        if group["quality_metric"] == "f1_macro"
                        else "Accuracy",
                        "quality_sd": "Quality SD",
                        "model": "Model",
                        "runs": "Runs",
                        "qualified": "Qualified",
                        "frontier": "Pareto frontier",
                        "memory_mib": "Max peak memory (MiB)",
                        "latency_ms": "Latency (ms)",
                        "training_time_sec": "Train (s)",
                    }
                )
            )
            sections.append(candidate_table)
            repeated = {
                c["model"]
                for c in candidates
                if sum(x["model"] == c["model"] for x in candidates) > 1
            }
            names = {
                c["candidate"]: model_name(c["model"])
                + (f" (configuration {i + 1})" if c["model"] in repeated else "")
                for i, c in enumerate(candidates)
            }
            recommendations = (
                group["recommendations"] if sum(c["qualified"] for c in candidates) > 1 else {}
            )
            for label, ids in recommendations.items():
                label = label.replace("_", " ").capitalize()
                chosen = ", ".join(names[i] for i in ids)
                sections.append(
                    f"<p><strong>{html.escape(label)}:</strong> {html.escape(chosen)}</p>"
                )
            for c in candidates:
                if c["constraint_failures"]:
                    reason = "; ".join(c["constraint_failures"])
                    sections.append(
                        f"<p>{html.escape(names[c['candidate']])}: {html.escape(reason)}</p>"
                    )
            try:
                paths = plot_pareto(group, output / group["group"])
                chart_paths.extend(paths)
                for path in paths:
                    add_plot(path, "Pareto Comparison")
            except (ValueError, ImportError) as exc:
                notices.append(f"Pareto plot: {exc}")
            # Build one row per tested configuration, without merging distinct recipes.
            frames = []
            for c in candidates:
                saved = [lookup[i] for i in c["run_ids"] if i in lookup]
                view = BenchmarkResults(saved)
                view.aggregate_splits = True
                frame = view.summary()
                if len(frame) == 1:
                    frame["model"] = names[c["candidate"]]
                    frames.append(frame)
            if frames:
                try:
                    paths = plot_summary(
                        pd.concat(frames, ignore_index=True), output / group["group"]
                    )
                    chart_paths.extend(paths)
                    for path in paths:
                        if "nodes_vs" not in path.name:
                            add_plot(path, path.stem.replace("_", " "))
                    if not any("gpu_memory" in p.name for p in paths):
                        sections.append(
                            "<p>GPU allocation chart unavailable: "
                            "these runs have no GPU measurements.</p>"
                        )
                except (ValueError, ImportError) as exc:
                    notices.append(f"Comparison plots: {exc}")
        failed_models = {r.get("model") for r in original_records if r.get("status") != "SUCCESS"}
        exclusions = {
            str(x.get("model")) + ": " + x["reason"]
            for x in group["excluded"]
            if x.get("model") not in failed_models
        }
        for reason in sorted(exclusions):
            sections.append(f"<p>{html.escape(reason)}</p>")
    if not analysis["groups"]:
        sections.append("<p>No verifiable comparison groups available.</p>")
    try:
        paths = plot_training_time(curve_records, output / "scaling")
        chart_paths.extend(paths)
        sections.append("<h2>Number of nodes vs training time</h2>")
        for path in paths:
            add_plot(path, "All available measured node counts; a single size is one point.")
    except (ValueError, ImportError) as exc:
        notices.append(f"Node-count plot: {exc}")
    for note in notices:
        sections.append(f'<p class="alert">{html.escape(note)}</p>')
    css = """.egb-analysis{font:15px/1.5 system-ui,sans-serif;max-width:1280px;margin:32px auto;
    padding:0 24px;color:#183247;background:#fff}
    .egb-analysis h1,.egb-analysis h2,.egb-analysis h3{color:#124d69}
    .egb-analysis table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px}
    .egb-analysis th,.egb-analysis td{padding:9px;border-bottom:1px solid #dce5eb;
    text-align:right;overflow-wrap:anywhere}
    .egb-analysis th:first-child,.egb-analysis td:first-child{text-align:left}
    .egb-analysis table thead th{background:#dcebf2!important;color:#163b50!important}
    .egb-analysis table tbody td{color:#183247!important;background:#fff!important}
    .egb-analysis table tbody tr:nth-child(even) td{background:#f8fafc!important}
    .egb-analysis img{width:100%;height:auto}.egb-analysis figure{margin:24px 0}
    .egb-analysis figcaption,.egb-analysis .note{color:#516779}
    .egb-analysis .criteria{background:#eef6fa;padding:16px 20px;border-left:4px solid #187399}
    .egb-analysis .criteria h4{margin:0 0 10px;color:#124d69;font-size:17px}
    .egb-analysis .criteria li{margin:10px 0}
    .egb-analysis .alert{background:#fff3cf;padding:10px}
    @media print{.egb-analysis figure,.egb-analysis table{break-inside:avoid}}"""
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>EfficientGraphBench analysis</title>"
    )
    from efficientgraphbench.execution_analysis.report import report_link
    from efficientgraphbench.results.theme import dark_css

    sections.append(report_link(results.records, output))
    css += dark_css(".egb-analysis")
    fragment = f'<style>{css}</style><div class="egb-analysis">' + "\n".join(sections) + "</div>"
    document += "</head><body>" + fragment + "</body></html>"
    report = output / "analysis.html"
    report.write_text(document, encoding="utf-8")
    notebook = False
    if display:
        try:
            from IPython import get_ipython
            from IPython.display import HTML
            from IPython.display import display as ipython_display

            notebook = (
                get_ipython() is not None and getattr(get_ipython(), "kernel", None) is not None
            )
        except ImportError:
            notebook = False
        if notebook:
            ipython_display(HTML(fragment))
        else:
            results.show()
            from rich.console import Console

            console = Console()
            for item in failures:
                console.print(
                    f"{item['Model']}: {item['Status']} — {item['Reason']} "
                    f"({item['Skipped']} skipped)",
                    markup=False,
                )
            for group in analysis["groups"]:
                for note in group_setup_notes(group, analysis["groups"]):
                    console.print(note, markup=False)
                console.print("Selection criteria", style="bold")
                for title, explanation in comparison_criteria(group):
                    console.print(title + ": " + explanation, markup=False)
                selected = group["candidates"]
                if selected:
                    from rich.table import Table

                    table = Table(
                        title=f"All eligible candidates ({group['num_nodes']} nodes)",
                        header_style="bold #163b50 on #dcebf2",
                    )
                    for label in ("Model", "Quality", "Latency ms", "Train s", "Memory MiB"):
                        table.add_column(label)
                    for c in selected:
                        table.add_row(
                            model_name(c["model"]),
                            f"{c['quality']:.2%}",
                            f"{c['latency_ms']:.3f}",
                            f"{c['training_time_sec']:.3f}",
                            f"{c['memory_mib']:.1f}" if c["memory_mib"] is not None else "—",
                        )
                    console.print(table)
                    labels = {c["candidate"]: model_name(c["model"]) for c in selected}
                    recommendations = (
                        group["recommendations"]
                        if sum(c["qualified"] for c in selected) > 1
                        else {}
                    )
                    for label, ids in recommendations.items():
                        console.print(
                            label.replace("_", " ").capitalize()
                            + ": "
                            + ", ".join(labels[i] for i in ids),
                            markup=False,
                        )
            for note in notices:
                console.print(note, markup=False)
            console.print(f"Open all tables and charts: {report.resolve()}", markup=False)
    if open_browser:
        import webbrowser

        webbrowser.open(report.resolve().as_uri())
    from efficientgraphbench.results.artifact import ReportOutput

    return ReportOutput(
        report=report,
        pareto=analysis,
        plots=chart_paths,
        notices=notices,
        notebook_displayed=notebook,
    )


def plot_pareto(group, output_dir):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    candidates = group["candidates"]
    if not candidates:
        return []
    fig = Figure(figsize=(9, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for i, c in enumerate(candidates):
        ax.errorbar(
            c["latency_ms"],
            c["quality"] * 100,
            yerr=c["quality_sd"] * 100,
            fmt="D" if c["frontier"] else "o",
            markersize=8,
            capsize=3,
            alpha=1 if c["qualified"] else 0.35,
            label=model_name(c["model"])
            + (
                f" (configuration {i + 1})"
                if sum(x["model"] == c["model"] for x in candidates) > 1
                else ""
            )
            + (" · frontier" if c["frontier"] else ""),
        )
    ax.set(
        xlabel="Inference latency (ms; lower is better)",
        ylabel=("Macro-F1" if group["quality_metric"] == "f1_macro" else "Accuracy")
        + " (%; higher is better)",
        title="Pareto Comparison",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.text(
        0.1,
        -0.06,
        "Diamonds: frontier across ALL selected objectives, including memory if selected.\n"
        "This is a 2D projection, not a fitted frontier curve. Bars: quality sample SD.",
        fontsize=9,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    from efficientgraphbench.results.plots import _dark_figure

    _dark_figure(fig)
    for suffix in ("png", "pdf"):
        path = output / f"pareto_frontier.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths.append(path)
    return paths
