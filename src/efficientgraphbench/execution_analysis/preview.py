"""A navigable execution report for notebook previews that block JavaScript."""

from collections import defaultdict
from html import escape

from efficientgraphbench.names import model_name


def _text(value):
    return escape(str(value if value is not None else "Unavailable"))


def _seconds(value):
    return "—" if value is None else f"{value:.3f}"


def preview(sessions):
    groups = defaultdict(lambda: defaultdict(list))
    for session in sessions:
        if session.get("model"):
            groups[session["model"]][session.get("seed")].append(session)
    out = [
        '<section id="safePreview"><h2>Execution results</h2>',
        "<p>Expand a model, then a seed and stage to inspect its saved measurements. "
        "This view works without scripts. For the interactive timeline and kernel filters, "
        "download and extract the <a href='execution-bundle.zip' download>report ZIP</a> "
        "and open execution.html in your browser. "
        "<a href='training-clock.csv' download>Download epoch and stage timings (CSV)</a>.</p>",
    ]
    if not groups:
        out.append("<p>No model execution sessions were recorded.</p>")
    for i, (model, seeds) in enumerate(groups.items()):
        out.append(
            f"<details{' open' if i == 0 else ''}><summary>"
            f"{_text(model_name(model))} · {len(seeds)} seed(s)</summary>"
        )
        for j, (seed, stages) in enumerate(seeds.items()):
            out.append(
                f"<details{' open' if i == j == 0 else ''}><summary>Seed {_text(seed)}</summary>"
            )
            for k, s in enumerate(stages):
                out.append(
                    f"<details{' open' if i == j == k == 0 else ''}><summary>"
                    f"{_text(s.get('stage_name'))} · {_text(s.get('origin'))} · "
                    f"{_text(s.get('status'))}</summary>"
                )
                out.append(
                    f"<p>Dataset: {_text(s.get('dataset'))} · "
                    f"Training loop: {_seconds(s.get('training_seconds'))} s · "
                    f"Whole stage: {_seconds(s.get('stage_wall_seconds'))} s</p>"
                )
                if s.get("failure_reason"):
                    out.append(f"<p>{_text(s['failure_reason'])}</p>")
                out.append(f"<p>{_text(s.get('profiling_status'))}</p>")
                for a in s.get("profile_attempts", []):
                    out.append(
                        f"<p>Epoch {_text(a.get('epoch'))} · GPU profiling: "
                        f"{_text(a.get('gpu_profiling_status') or a.get('status'))} · "
                        f"Kernel events: {_text(a.get('kernel_events'))}. "
                        f"{_text(a.get('cuda_capture_reason') or a.get('reason') or '')}</p>"
                    )
                rows = s.get("epoch_clock", [])
                if rows:
                    out.append(
                        "<details><summary>Epoch timings</summary>"
                        "<pre>Epoch     Wall (s)     Optimization (s)"
                        "     Validation (s)     Status\n"
                    )
                    for r in rows:
                        out.append(
                            f"{_text(r['epoch']):>5} {_seconds(r['epoch_wall_seconds']):>12} "
                            f"{_seconds(r['optimization_seconds']):>20} "
                            f"{_seconds(r['validation_seconds']):>18}     "
                            f"{_text(r['status'])}\n"
                        )
                    out.append("</pre></details>")
                out.append("</details>")
            out.append("</details>")
        out.append("</details>")
    return "".join(out) + "</section>"
