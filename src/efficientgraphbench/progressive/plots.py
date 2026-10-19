"""Small static report figures derived only from successful measured records."""

import base64
import io


def report_figures(records, compute):
    import matplotlib

    matplotlib.use("Agg")
    from statistics import mean

    import matplotlib.pyplot as plt
    import numpy as np

    from efficientgraphbench.names import model_name

    good = [r for r in records if r.get("status") == "SUCCESS"]
    if not good:
        return ""
    models = list(dict.fromkeys(r["model"] for r in good))
    stages = sorted({r["fidelity"]["stage_index"] for r in good})
    colors = [
        "#60A5FA",
        "#22D3EE",
        "#34D399",
        "#FBBF24",
        "#FB7185",
        "#C4B5FD",
        "#FDBA74",
        "#F0ABFC",
    ]
    figures = []

    def finish(fig, ax, title, ylabel):
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        ax.set_title(title, color="#E8EEF7", loc="left", fontsize=12, pad=14)
        ax.set_ylabel(ylabel, color="#B8C5D9")
        ax.tick_params(colors="#B8C5D9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#293548")
        ax.grid(axis="y", color="#293548", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                facecolor="#172033",
                labelcolor="#E8EEF7",
                edgecolor="#293548",
                fontsize=7,
                ncol=min(4, len(handles)),
            )
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        figures.append(
            '<img class="chart" alt="'
            + title
            + '" src="data:image/png;base64,'
            + base64.b64encode(stream.getvalue()).decode()
            + '">'
        )

    fig, ax = plt.subplots(figsize=(7, 3.1))
    for j, model in enumerate(models):
        rows = [r for r in good if r["model"] == model]
        xs = sorted({r["fidelity"]["stage_index"] for r in rows})
        ax.plot(
            [x + 1 for x in xs],
            [
                mean(r["validation_metric"] for r in rows if r["fidelity"]["stage_index"] == x)
                for x in xs
            ],
            marker="o",
            color=colors[j % len(colors)],
            label=model_name(model),
        )
    ax.set_xticks([i + 1 for i in stages])
    ax.set_xlabel("Fidelity stage — populations grow", color="#B8C5D9")
    finish(fig, ax, "Validation across stages", "Mean validation score")

    transitions = [
        r for r in good if r.get("adaptation", {}).get("new_node_score_pre_training") is not None
    ]
    if transitions:
        last = max(r["fidelity"]["stage_index"] for r in transitions)
        rows = [r for r in transitions if r["fidelity"]["stage_index"] == last]
        names = list(dict.fromkeys(r["model"] for r in rows))
        xs = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7, 3.1))
        for j, (key, label) in enumerate(
            [
                ("old_node_score_pre_training", "Old before"),
                ("old_node_metric", "Old after"),
                ("new_node_score_pre_training", "New before"),
                ("new_node_metric", "New after"),
            ]
        ):
            ax.bar(
                xs + (j - 1.5) * 0.19,
                [mean(r["adaptation"][key] for r in rows if r["model"] == m) for m in names],
                width=0.18,
                color=colors[j],
                label=label,
            )
        ax.set_xticks(xs, [model_name(m) for m in names])
        finish(fig, ax, f"Stage {last + 1}: same-node pre/post adaptation", "Mean validation score")

    if any("optimization_seconds" in r for r in good):
        fig, ax = plt.subplots(figsize=(7, 2.7))
        bottom = np.zeros(len(stages))
        for j, (keys, label) in enumerate(
            [
                (["optimization_seconds"], "Optimization"),
                (
                    [
                        "validation_seconds",
                        "pre_adaptation_evaluation_seconds",
                        "post_adaptation_evaluation_seconds",
                        "test_evaluation_seconds",
                    ],
                    "Evaluation",
                ),
            ]
        ):
            vals = [
                sum(
                    sum(r.get(k, 0) for k in keys)
                    for r in good
                    if r["fidelity"]["stage_index"] == i
                )
                for i in stages
            ]
            ax.bar([i + 1 for i in stages], vals, bottom=bottom, color=colors[j], label=label)
            bottom += vals
        ax.set_xticks([i + 1 for i in stages])
        ax.set_xlabel("Fidelity stage", color="#B8C5D9")
        finish(fig, ax, "Measured execution cost by stage", "Seconds — sum across seeds/models")
    if compute.get("full_benchmark_elapsed_seconds") is not None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.bar(
            ["Full", "Progressive"],
            [compute["full_benchmark_elapsed_seconds"], compute["progressive_elapsed_seconds"]],
            color=colors[:2],
        )
        finish(fig, ax, "Compatible measured runtime comparison", "Total elapsed seconds")
    return "".join(figures)
