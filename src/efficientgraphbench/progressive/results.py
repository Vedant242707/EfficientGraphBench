"""Separate, readable progressive reports; partial graphs never share finalist tables."""

import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from efficientgraphbench.names import model_name
from efficientgraphbench.progressive.adaptation import transition_metrics
from efficientgraphbench.progressive.plots import report_figures
from efficientgraphbench.results.protocol import validate_progressive_records
from efficientgraphbench.results.theme import dark_css

GLOSSARY = [
    (
        "Old-node performance delta",
        "old post-training − previous-stage score",
        "Positive: improvement; negative: degradation on the same old validation nodes.",
    ),
    (
        "Old-node forgetting",
        "max(0, previous-stage score − old post-training)",
        "Non-negative loss relative to the previous stage; zero includes improvement.",
    ),
    (
        "Post-expansion delta",
        "old pre-training − previous-stage score",
        "Positive: graph expansion helped old nodes; negative: it harmed them before adaptation.",
    ),
    (
        "New-node adaptation gain",
        "new post-training − new pre-training",
        "Positive: additional training improved predictions on new validation nodes.",
    ),
    (
        "Old-node adaptation delta",
        "old post-training − old pre-training",
        "Positive: adaptation helped old nodes; negative: adaptation harmed them.",
    ),
    (
        "Old/new gap",
        "abs(old post-training − new post-training)",
        "Diagnostic only: a small gap can also mean both scores are poor.",
    ),
    (
        "Recovery status",
        "old validation score compared with threshold × previous-stage score",
        "recovery_not_required: target met before training; recovered: met during training; "
        "not_recovered: not met within budget; not_applicable: no valid reference.",
    ),
    (
        "Recovery epochs / seconds",
        "first additional epoch / elapsed training time reaching target",
        "Zero means recovery_not_required. Recovery means below exclude those "
        "zero-cost transitions.",
    ),
    (
        "Training-phase difference (%) — validation policies differ",
        "100 × (full training − progressive training) / full training",
        "Positive: less training time; negative: more training time. Requires "
        "a compatible baseline.",
    ),
    (
        "Wall-clock reduction (%)",
        "100 × (full elapsed − progressive elapsed) / full elapsed",
        "Includes overhead. Negative values mean progressive execution was slower.",
    ),
]


def adaptation(row: dict) -> dict:
    """Read previous records without mutating them or inventing missing measurements."""

    a = dict(row.get("adaptation", {}))

    if "previous_stage_score" not in a:
        a.update(
            transition_metrics(
                a.get("performance_before_expansion"),
                a.get("performance_after_expansion_before_adaptation"),
                None,
                a.get("old_node_metric"),
                a.get("new_node_metric"),
            )
        )

        a["recovery_epochs"] = a.get("epochs_to_recover")

        a["recovery_seconds"] = a.get("time_to_recover_seconds")

        if a.get("recovery_status") == "recovered" and a.get("recovery_epochs") == 0:
            a["recovery_status"] = "recovery_not_required"

    return a


def average(rows: list[dict], key: str) -> float | None:

    values = [adaptation(r).get(key) for r in rows]

    values = [v for v in values if v is not None]

    return mean(values) if values else None


def recovery_counts(rows: list[dict]) -> str:

    counts = Counter(adaptation(r).get("recovery_status", "not_applicable") for r in rows)

    return "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in sorted(counts.items()))


def recovery_summary(rows):

    values = [adaptation(r) for r in rows]

    required = [a for a in values if a.get("recovery_status") in {"recovered", "not_recovered"}]

    return {
        "Transitions requiring recovery": len(required),
        "Recovered transitions": sum(a.get("recovery_status") == "recovered" for a in required),
        "Never recovered": sum(a.get("recovery_status") == "not_recovered" for a in required),
        "Final target satisfied": sum(a.get("final_target_satisfied") is True for a in values),
        "Final target evaluated": sum(a.get("final_target_satisfied") is not None for a in values),
    }


def render_table(frame: pd.DataFrame) -> str:

    if frame.empty:
        return "<p>None.</p>"

    if len(frame.columns) > 6:
        anchor = frame.columns[0]

        return "".join(
            render_table(frame[[anchor, *frame.columns[i : i + 5]]])
            for i in range(1, len(frame.columns), 5)
        )

    text_columns = {
        "Model",
        "State",
        "Reason",
        "Recovery status",
        "Setting",
        "Value",
        "Metric",
        "Formula",
        "Meaning",
        "Outcome",
    }

    parts = ['<div class="table-scroll"><table><thead><tr>']

    align = ["text" if c in text_columns else "numeric" for c in frame.columns]

    for col, cls in zip(frame.columns, align):
        parts.append(f'<th scope="col" class="{cls}">{html.escape(str(col))}</th>')

    parts.append("</tr></thead><tbody>")

    for row in frame.itertuples(index=False, name=None):
        parts.append("<tr>")

        for value, cls, column in zip(row, align, frame.columns):
            if isinstance(value, (int, float)) and any(
                term in column.lower() for term in ("delta", "gain")
            ):
                cls += " positive" if value > 0 else " negative" if value < 0 else ""

            value = (
                "—"
                if pd.isna(value)
                else f"{value:.3f}"
                if isinstance(value, float)
                else str(value)
            )

            parts.append(f'<td class="{cls}">{html.escape(value)}</td>')

        parts.append("</tr>")

    return "".join(parts) + "</tbody></table></div>"


class ProgressiveResults:
    def __init__(self, records: list[dict], output: Path, metadata: dict):
        validate_progressive_records(records)
        self.records, self.output_dir, self.metadata = records, Path(output), metadata

    def summary(self) -> pd.DataFrame:
        validate_progressive_records(self.records)
        rows = []

        for model in dict.fromkeys(r["model"] for r in self.records):
            records = [r for r in self.records if r["model"] == model]

            last = max(r["fidelity"]["stage_index"] for r in records)

            attempts = [r for r in records if r["fidelity"]["stage_index"] == last]

            good = [r for r in records if r["status"] == "SUCCESS"]

            final = [r for r in attempts if r["status"] == "SUCCESS"]

            finalist = (
                bool(final)
                and len(final) == len(self.metadata.get("seeds", attempts))
                and all(
                    r["fidelity"]["graph_fraction"] == 1 and r["candidate_status"] == "advanced"
                    for r in attempts
                )
            )

            scores = [r["validation_metric"] for r in final]

            recovered = [r for r in good if adaptation(r).get("recovery_status") == "recovered"]

            row = {
                "Model": model_name(model),
                "Finalist": finalist,
                "State": "completed" if finalist else attempts[-1]["candidate_status"],
                "Last stage": last + 1,
                "Graph fraction": attempts[0]["fidelity"]["graph_fraction"],
                "Validation mean": mean(scores) if scores else None,
                "Validation SD": stdev(scores) if len(scores) > 1 else 0.0 if scores else None,
                "Successful seeds": len(final),
                "Training s (all seeds)": sum(r.get("training_time_sec", 0) for r in good),
                "Reason": next(
                    (r.get("failure_reason") for r in attempts if r.get("failure_reason")),
                    attempts[0].get("promotion", {}).get("reason", "not recorded"),
                ),
                "Recovery status": recovery_counts(good),
                **recovery_summary(good),
                "Mean recovery epochs": average(recovered, "recovery_epochs"),
                "Mean recovery s": average(recovered, "recovery_seconds"),
            }

            for label, key in [
                ("Old validation", "old_node_metric"),
                ("New validation", "new_node_metric"),
                ("Old/new gap", "old_new_gap"),
                ("New adaptation gain", "new_node_adaptation_gain"),
                ("Old adaptation delta", "old_node_adaptation_delta"),
                ("Old-node delta", "old_node_performance_delta"),
                ("Forgetting", "old_node_forgetting"),
            ]:
                row[label] = average(final, key)

            for key in ("accuracy", "f1_macro", "f1_micro"):
                values = [
                    r["test_metrics"][key] for r in final if finalist and r.get("test_metrics")
                ]

                row["Test " + key] = mean(values) if values else None

            for label, key in [
                ("GPU allocated MiB", "peak_gpu_allocated_mib"),
                ("GPU reserved MiB", "peak_gpu_reserved_mib"),
                ("RAM MiB", "peak_cpu_memory_mb"),
            ]:
                values = [r[key] for r in good if r.get(key) is not None]

                row[label] = max(values) if values else None

            rows.append(row)

        return pd.DataFrame(rows)

    def execution_analysis(self, output_dir=None):
        from efficientgraphbench.execution_analysis import execution_report

        return execution_report(
            self.records, output_dir or self.output_dir / "execution-analysis", self.metadata
        )

    def save(self) -> Path:

        self.output_dir.mkdir(parents=True, exist_ok=True)

        path = self.output_dir / "progressive-results.json"

        path.write_text(
            json.dumps(
                {"metadata": self.metadata, "records": self.records}, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )

        self.summary().to_csv(self.output_dir / "progressive-summary.csv", index=False)

        return path

    def show_all(self, *, open_browser: bool = False, display: bool = True) -> Path:

        self.save()

        m, cfg = self.metadata, self.metadata.get("config", {})

        summary = self.summary()

        stages = cfg.get("stages", [])

        scale = m.get("dataset_scale", {})

        progression = [
            len({r["model"] for r in self.records if r["fidelity"]["stage_index"] == i})
            for i in range(len(stages))
        ]

        setup = {
            "Mode": "Progressive Multi-Fidelity — expanded transductive retraining",
            "Dataset": m.get("dataset", "not recorded"),
            "Dataset scale": ", ".join(f"{k}: {v:,}" for k, v in scale.items()) or "not recorded",
            "Progression": m.get(
                "progression_strategy", cfg.get("progression_strategy", "not recorded")
            ),
            "Graph fractions": " → ".join(f"{s['graph_fraction']:.0%}" for s in stages),
            "Stage nodes": " → ".join(str(n) for n in m.get("stage_node_counts", [])),
            "Additional epochs": " → ".join(f"+{s['additional_epochs']}" for s in stages),
            "Hardware": str(
                m.get("hardware", {}).get("gpu") or m.get("hardware", {}).get("cpu", "Not recorded")
            ),
            "Models selected": ", ".join(model_name(v) for v in m.get("models", [])),
            "Checkpoint policy": "Resume model + optimizer + scheduler (when configured)",
            "Selection metric": cfg.get("selection_metric", "not recorded").replace("_", " "),
            "Seeds": ", ".join(str(v) for v in m.get("seeds", [])),
            "Candidates attempted per stage": " → ".join(str(v) for v in progression),
            "Pruning safeguard": f"Margin {cfg.get('promotion_margin', 0):.3f}; "
            f"uncertainty protection: {cfg.get('uncertainty_aware', False)}",
        }

        parts = [
            "<h1>Progressive Multi-Fidelity Benchmark</h1>",
            '<div class="cards">'
            + "".join(
                f'<div class="card"><b>{value}</b><span>{label}</span></div>'
                for label, value in [
                    ("Nodes", scale.get("nodes", "Unavailable")),
                    ("Selected models", len(m.get("models", []))),
                    ("Seeds", len(m.get("seeds", []))),
                ]
            )
            + "</div>",
            "<p>This protocol continues training as the graph grows. It is not equivalent to "
            "the exhaustive Static Full Benchmark. Scores are fractions (0.80 = 80%); "
            "deltas are after − before: positive means improvement. A dash means unavailable.</p>",
            render_table(pd.DataFrame(setup.items(), columns=["Setting", "Value"])),
            '<div class="stage-flow">'
            + "".join(
                f'<div class="stage-box"><b>{stage["graph_fraction"]:.0%} graph</b>'
                f"<br>+{stage['additional_epochs']} epochs<br>{progression[i]} candidates</div>"
                for i, stage in enumerate(stages)
            )
            + "</div>",
            (
                "<h2>Full-graph finalists</h2><p>Scores: mean across final-stage "
                "seeds. Time: sum across seeds/stages. Memory: maximum across "
                "seeds/stages. Recovery: all stage transitions; means use "
                "recovered transitions only.</p>"
            ),
        ]

        finalists = summary[summary["Finalist"]] if not summary.empty else summary

        partial = summary[~summary["Finalist"]] if not summary.empty else summary

        groups = [
            [
                "Model",
                "Successful seeds",
                "Validation mean",
                "Validation SD",
                "Test accuracy",
                "Test f1_macro",
                "Test f1_micro",
            ],
            [
                "Model",
                "Old validation",
                "New validation",
                "Old/new gap",
                "New adaptation gain",
                "Old adaptation delta",
                "Old-node delta",
                "Forgetting",
            ],
            [
                "Model",
                "Recovery status",
                "Mean recovery epochs",
                "Mean recovery s",
                "Training s (all seeds)",
                "GPU allocated MiB",
                "GPU reserved MiB",
                "RAM MiB",
            ],
        ]

        groups.append(
            [
                "Model",
                "Transitions requiring recovery",
                "Recovered transitions",
                "Never recovered",
                "Final target satisfied",
                "Final target evaluated",
            ]
        )

        for columns in groups:
            parts.append(render_table(finalists.reindex(columns=columns)))

        parts += [
            "<h2>Early-eliminated candidates</h2>",
            "<p>Partial-graph scores below are history, not comparable final "
            "full-graph results.</p>",
        ]

        columns = [
            "Model",
            "Last stage",
            "Graph fraction",
            "Validation mean",
            "Validation SD",
            "Successful seeds",
            "Training s (all seeds)",
            "Reason",
        ]

        eliminated = partial[partial["State"] == "eliminated"] if not partial.empty else partial

        failed = partial[partial["State"] != "eliminated"] if not partial.empty else partial

        parts.append(render_table(eliminated.reindex(columns=columns)))

        parts += [
            "<h3>Failed, skipped or unsupported candidates</h3>",
            render_table(failed.reindex(columns=["Model", "State", *columns[1:]])),
        ]

        for index in sorted({r["fidelity"]["stage_index"] for r in self.records}):
            records = [r for r in self.records if r["fidelity"]["stage_index"] == index]

            f = records[0]["fidelity"]

            parts.append(
                f"<h2>Stage {index + 1}: {html.escape(f['stage_name'])}</h2>"
                f"<p>Graph {f['graph_fraction']:.0%}; "
                f"+{f['additional_epochs']} epochs per seed. "
                "Scores: seed means; time: sum across seeds; memory: maximum across seeds.</p>"
            )

            rows = []

            for model in dict.fromkeys(r["model"] for r in records):
                attempts = [r for r in records if r["model"] == model]

                good = [r for r in attempts if r["status"] == "SUCCESS"]

                scores = [r["validation_metric"] for r in good]

                item = {
                    "Model": model_name(model),
                    "State": attempts[-1]["candidate_status"],
                    "Nodes": attempts[0].get("num_nodes"),
                    "Edges": attempts[0].get("num_edges"),
                    "Successful seeds": len(good),
                    "Validation mean": mean(scores) if scores else None,
                    "Validation SD": stdev(scores) if len(scores) > 1 else 0.0 if scores else None,
                    "Recovery status": recovery_counts(good),
                    **recovery_summary(good),
                    "Training s": sum(r["training_time_sec"] for r in good),
                }

                metrics = {
                    "Previous score": "previous_stage_score",
                    "Old before": "old_node_score_pre_training",
                    "New before": "new_node_score_pre_training",
                    "Old after": "old_node_metric",
                    "New after": "new_node_metric",
                    "Expansion delta": "post_expansion_delta",
                    "Old adaptation": "old_node_adaptation_delta",
                    "New gain": "new_node_adaptation_gain",
                    "Old-node delta": "old_node_performance_delta",
                    "Forgetting": "old_node_forgetting",
                    "Old/new gap": "old_new_gap",
                    "Recovery epochs": "recovery_epochs",
                    "Recovery s": "recovery_seconds",
                }

                item.update({label: average(good, key) for label, key in metrics.items()})

                recovered = [r for r in good if adaptation(r).get("recovery_status") == "recovered"]

                item["Recovery epochs"] = average(recovered, "recovery_epochs")

                item["Recovery s"] = average(recovered, "recovery_seconds")

                for label, key in [
                    ("GPU peak MiB", "peak_gpu_allocated_mib"),
                    ("RAM peak MiB", "peak_cpu_memory_mb"),
                ]:
                    values = [r[key] for r in good if r.get(key) is not None]

                    item[label] = max(values) if values else None

                item["Reason"] = next(
                    (r.get("failure_reason") for r in attempts if r.get("failure_reason")),
                    attempts[0].get("promotion", {}).get("reason", "not recorded"),
                )

                rows.append(item)

            frame = pd.DataFrame(rows)

            parts.append(
                render_table(
                    frame.reindex(
                        columns=[
                            "Model",
                            "Transitions requiring recovery",
                            "Recovered transitions",
                            "Never recovered",
                            "Final target satisfied",
                            "Final target evaluated",
                        ]
                    )
                )
            )

            if index < len(m.get("selection_counts", [])):
                parts.append(
                    render_table(
                        pd.DataFrame(
                            [
                                (k.replace("_", " "), v)
                                for k, v in m["selection_counts"][index].items()
                            ],
                            columns=["Setting", "Value"],
                        )
                    )
                )

            for cols in [
                [
                    "Model",
                    "State",
                    "Nodes",
                    "Edges",
                    "Successful seeds",
                    "Validation mean",
                    "Validation SD",
                    "Reason",
                ],
                ["Model", "Previous score", "Old before", "New before", "Old after", "New after"],
                [
                    "Model",
                    "Expansion delta",
                    "Old adaptation",
                    "New gain",
                    "Old-node delta",
                    "Forgetting",
                    "Old/new gap",
                ],
                [
                    "Model",
                    "Recovery status",
                    "Recovery epochs",
                    "Recovery s",
                    "Training s",
                    "GPU peak MiB",
                    "RAM peak MiB",
                ],
            ]:
                parts.append(render_table(frame.reindex(columns=cols)))

        parts.append("<h2>Adaptation and resource visualizations</h2>")

        parts.append(report_figures(self.records, m.get("compute", {})))

        compute = m.get("compute", {})

        labels = {
            "full_benchmark_baseline": "Full benchmark baseline",
            "baseline_reason": "Baseline check",
            "full_benchmark_training_seconds": "Full training (s)",
            "progressive_training_seconds": "Progressive training (s)",
            "full_benchmark_elapsed_seconds": "Full elapsed (s)",
            "progressive_elapsed_seconds": "Progressive elapsed (s)",
            "optimization_seconds": "Optimization (s) — sum across attempts",
            "evaluation_seconds": "Evaluation (s) — sum across attempts",
            "other_execution_seconds": "Other execution (s) — elapsed remainder, not wasted time",
            "failed_attempt_elapsed_seconds": (
                "Failed attempt elapsed (s) — included above; do not add"
            ),
            "full_optimization_seconds": "Full optimization (s)",
            "optimization_reduction_percent": "Optimization reduction (%) — matching scope",
            "training_time_reduction_percent": (
                "Training-phase difference (%) — validation policies differ"
            ),
            "wall_clock_reduction_percent": "Wall-clock reduction (%)",
            "status": "Outcome",
        }

        parts += [
            "<h2>Resource analysis</h2>",
            render_table(
                pd.DataFrame(
                    [(label, compute.get(key)) for key, label in labels.items()],
                    columns=["Setting", "Value"],
                )
            ),
            "<h2>Methodology and metric glossary</h2>",
            "<p>Full uses per-epoch validation and best-validation weights. Progressive "
            "uses last-stage weights, stage-end validation and recovery checks until the "
            "first threshold crossing. Equal epochs do not mean equal graph exposure or "
            "compute; differences cannot be attributed solely to graph expansion.</p>",
            render_table(pd.DataFrame(GLOSSARY, columns=["Metric", "Formula", "Meaning"])),
            "<p>Selection uses validation scores only. Sample standard deviation "
            "describes seed variability, "
            "not statistical significance. Margin or overlapping mean ± SD bands "
            "may retain extra candidates. "
            "Final test labels never select candidates. Recovery averages "
            "include recovered transitions only. "
            "Finalist recovery statistics pool all transitions; stage "
            "statistics cover that stage only. "
            "First threshold crossing does not establish sustained recovery; "
            "final target satisfaction is recorded separately. "
            "Recovery target is the configured fraction of previous-stage overall "
            "validation, evaluated "
            "on those same old nodes after expansion.</p>",
            "<p>Memory is a per-stage allocator/process peak, not summed across "
            "stages or total device VRAM. "
            "RSS includes libraries and retained allocations. Training time "
            "includes validation checks; "
            "elapsed time includes initial report generation but excludes the "
            "final metadata refresh. "
            "CUDA can be nondeterministic despite fixed seeds. Efficiency conclusions apply to the "
            "reported dataset size and hardware; small-graph results do not "
            "establish large-graph savings.</p>",
        ]

        css = "<style>" + dark_css(".progressive") + "</style>"
        from efficientgraphbench.execution_analysis.report import report_link

        parts.append(report_link(self.records, self.output_dir, self.metadata))
        fragment = css + '<div class="progressive">' + "\n".join(parts) + "</div>"

        path = self.output_dir / "progressive-report.html"

        path.write_text('<!doctype html><meta charset="utf-8">' + fragment, encoding="utf-8")

        if display:
            shown = False

            try:
                from IPython import get_ipython
                from IPython.display import HTML
                from IPython.display import display as notebook_display

                if getattr(get_ipython(), "kernel", None) is not None:
                    notebook_display(HTML(fragment))

                    shown = True

            except ImportError:
                pass

            if not shown:
                print("FULL-GRAPH FINALISTS\n" + finalists.to_string(index=False))

                print("ELIMINATED / INCOMPLETE CANDIDATES\n" + partial.to_string(index=False))

            print(f"Progressive report: {path.resolve()}")

        if open_browser:
            import webbrowser

            webbrowser.open(path.resolve().as_uri())

        return path
