"""Standalone score explorer; native disclosure controls work in Jupyter previews."""

from collections import defaultdict
from html import escape

from efficientgraphbench.names import model_name


def score_dashboard(records, output, *, progressive=False):
    mode = "Progressive" if progressive else "Full"
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[str(r.get("model", "Unknown"))][str(r.get("seed", "—"))].append(r)

    def text(value):
        return escape(str(value)) if value is not None else "—"

    def percent(value):
        return f"{value:.2%}" if isinstance(value, (int, float)) else "—"

    def seconds(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "—"

    title = f"{mode} benchmark score dashboard"
    body = [
        '<!doctype html><html><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1">',
        f"<title>{title}</title><style>body{{font:16px/1.5 system-ui;margin:2rem;"
        "background:#101827;color:#eaf1fb}details{margin:1em 0;padding:.6em 1em;"
        "background:#1c283a;border:1px solid #52627a}summary{cursor:pointer;font-weight:600}"
        "table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.5em;"
        "border-bottom:1px solid #52627a}th{background:#273c56;color:white}"
        ".scroll{overflow:auto}</style>",
        f"<h1>{title}</h1><p>Expand a model and seed to inspect measured scores. "
        "A dash means the measurement is unavailable.</p>",
    ]
    if progressive:
        body.append(
            "<p>Validation is measured at each stage. Test scores appear only when "
            "recorded at the final stage.</p>"
        )
    if not groups:
        body.append("<p>No benchmark records available.</p>")
    for index, (model, seeds) in enumerate(groups.items()):
        body.append(
            f"<details{' open' if index == 0 else ''}><summary>"
            f"{text(model_name(model))} · {len(seeds)} seed(s)</summary>"
        )
        for j, (seed, runs) in enumerate(seeds.items()):
            body.append(
                f"<details{' open' if index == j == 0 else ''}><summary>Seed {text(seed)}"
                '</summary><div class="scroll"><table><thead><tr>'
            )
            headers = (
                ["Dataset", "Stage", "Status", "Candidate", "Validation score"]
                if progressive
                else ["Dataset", "Status", "Validation accuracy"]
            )
            headers += ["Test accuracy", "Macro-F1", "Micro-F1", "Training (s)", "Reason"]
            body.extend(f"<th>{h}</th>" for h in headers)
            body.append("</tr></thead><tbody>")
            for r in runs:
                status = r.get("status")
                metrics = (r.get("test_metrics") or {}) if progressive else r
                cells = [text(r.get("dataset"))]
                if progressive:
                    stage = r.get("fidelity") or {}
                    cells.append(text(stage.get("stage_name") or stage.get("stage_index")))
                cells.append(text(status))
                if progressive:
                    cells.append(text(r.get("candidate_status")))
                validation = r.get("validation_metric" if progressive else "validation_accuracy")
                cells.append(percent(validation) if status == "SUCCESS" else "—")
                cells.extend(
                    percent(metrics.get(k)) if status == "SUCCESS" else "—"
                    for k in ("accuracy", "f1_macro", "f1_micro")
                )
                cells.extend([seconds(r.get("training_time_sec")), text(r.get("failure_reason"))])
                body.append("<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
            body.append("</tbody></table></div></details>")
        body.append("</details>")
    path = output / "benchmark-dashboard.html"
    path.write_text("".join(body) + "</html>", encoding="utf8")
    return path
