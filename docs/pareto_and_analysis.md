# Selected candidates and combined results

After a benchmark, run one Python command:

```python
report = results.show_all()
```

In Jupyter this displays classification metrics, performance, unusual-result notes,
the macro/micro explanation, all-model statistics, Pareto recommendations and plots together. It also saves a
standalone `analysis/analysis.html` inside the experiment directory. Images are
embedded, so you can share that HTML alone. No retraining or dataset download occurs.

From a terminal, analyze saved results and open the report:

```bash
egbench analyze --input experiments/local-texas --open
```

For remote SSH, omit `--open` and copy the HTML to your laptop. The terminal prints
tables and selected candidates; the report displays the graphics.

Reload in Python without retraining:

```python
from efficientgraphbench.results.dashboard import load_analysis_results
results = load_analysis_results("experiments/local-texas")
report = results.show_all()
```

Inputs: experiment directory, run JSON/JSONL, or `scaling-results.json`. Scaling
reports use all sizes for the training-time curve and all seeds at each model's
last successful size for tables. Different graph sizes receive separate frontiers.
Models with no successes retain their failures. Missing metrics or verification
metadata exclude recommendations with a reason rather than inventing values.

## Frontier and recommendations

Default CUDA objectives: maximize mean Macro-F1, minimize mean per-run median
inference latency, and minimize maximum GPU reserved-memory peak across runs.
CPU auto mode uses quality and latency; it does not substitute CPU RSS or zero GPU
memory. Training time can be included as another objective.

Recommendations: best quality, fastest inference, fastest training, lowest memory
(when selected), and overall Pareto shortlist. Tied winners stay. Maintained-library
MLP is included as a feature-only baseline.

Qualified means meeting the selected limits, not necessarily having good accuracy.
Default score minima are zero and resource budgets are unset. A qualified candidate
is on the frontier unless another qualified candidate is at least as good on every
selected objective and strictly better on one. Equal objective values retain both.
This is descriptive dominance, not a statistical significance test.

Candidates are model configurations, not universal model identities. Graph content,
size, hardware, measurement settings, source snapshots, split mode and training
budget define separate groups. Model recipes/environments may differ within a group
and stay separate candidates. All matched seed/split attempts must succeed; failed,
skipped, duplicate or incomplete runs exclude a candidate. Quality SD describes
observed variability. A lone candidate is trivially on its frontier, without proving
superiority. The Pareto plot is a quality/latency projection of the full objective
set; memory can preserve a frontier candidate that looks dominated in 2D.

## Options

```python
report = results.show_all(
    minimum_quality=0.70,  # Illustrative Macro-F1 fraction, not percent
    minimum_accuracy=0.75,
    maximum_latency_ms=20,
    minimum_runs=10,
    include_training=True,
)
```

These are example constraints, not universal acceptance thresholds.

| Python argument / CLI option | Default | Meaning |
|---|---|---|
| `quality` / `--quality` | `f1_macro` | `f1_macro` or `accuracy` |
| `memory` / `--memory` | `auto` | `auto`, `gpu`, `cpu`, `none` |
| `include_training` / `--include-training` | False | Training time becomes an objective |
| `minimum_quality` / `--minimum-quality` | 0 | Required mean quality, fraction |
| `minimum_accuracy` / `--minimum-accuracy` | 0 | Required mean accuracy, fraction |
| `maximum_latency_ms` / `--maximum-latency-ms` | None | Mean latency limit |
| `maximum_memory_mib` / `--maximum-memory-mib` | None | Maximum run peak memory limit |
| `minimum_runs` / `--minimum-runs` | 1 | Required matched successful runs |
| `weights` (Python only) | None | Explicit priorities for optional balanced selection |

`memory="cpu"` uses process RSS, which includes libraries and allocator retention;
it is not isolated model memory, and execution routes differ. GPU recommendation
memory uses maximum reserved peaks; allocation charts show mean allocated peaks.

Optional balanced selection on CUDA without training as an objective:

```python
analysis = results.pareto(
    weights={"quality": 0.6, "latency_ms": 0.3, "memory_mib": 0.1}
)
```

Weights must cover all active objectives, be nonnegative and have positive sum.
Objectives are normalized across qualifying candidates to losses in [0,1]; constant
objectives contribute zero. The lowest weighted-loss frontier candidates are chosen.
This choice depends on the candidate set and supplied priorities. For CPU auto mode,
omit `memory_mib`; when including training, add `training_time_sec`. No hidden
weighted winner is chosen by default.

## Separate functions and commands

```python
results.show()                         # Metrics and notes
analysis = results.pareto()            # Structured Pareto analysis
table = results.summary(detailed=True) # Numeric export/analysis table
```

```bash
egbench pareto --input experiments/local-texas --output experiments/local-texas/pareto.json
egbench analyze --help
egbench pareto --help
```

`show_all(output_dir=..., display=False)` saves without displaying;
`show_all(open_browser=True)` opens the report. Install charts if needed with
`python -m pip install '.[plots]'` from the repository.

Outputs: `analysis.html`, `pareto.json`, `summary.csv`, PNG/PDF/CSV plot exports.
GPU allocation charts are unavailable on CPU. One measured node count gives one
point on the training-time plot. Missing plots carry explicit explanations.

## Presentation rules

Notes require at least five distinct runs and a pattern in at least 80% of eligible runs. A persistent Macro-F1/Micro-F1 gap must be at least 15 percentage points with at least 100 test nodes per run. Zero recall requires at least 20 test examples of that class per run. These are descriptive screening rules, not significance tests; scores remain visible even when support is too small for a note.

The combined report shows all eligible candidates, objective-specific recommendations and the Pareto chart. Explicit `results.pareto()` analysis remains available. Failed models appear once with the reason and skipped count; successful measurements remain visible, with a note when a model did not finish all attempts. Raw records and numeric summaries retain all attempts for inspection. Table headers use explicit dark text on a light background, including inside dark notebook themes.
