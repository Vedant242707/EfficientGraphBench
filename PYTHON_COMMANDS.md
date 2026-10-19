# EfficientGraphBench — notebook guide

Dataset statistics: `from efficientgraphbench.datasets import profile_dataset`, then `print(profile_dataset("cora"))`. See [Dataset profiling](docs/dataset_profiling.md).

Need accuracy/latency and GPU memory charts? See [Plot saved results](docs/plots.md). No retraining is required.

**Already installed? Start at Step 1. You do not need to run every example in this file.**

**College GPU / professor's experiments:** see [the short college guide](docs/college_experiments.md). Use `runs=10` or `runs=20` instead of `seeds=[...]`; use `splits="official"` to cycle heterophilous folds or `splits="random"` for new 60/20/20 splits.

Choose the notebook kernel using the project's Python environment. Restart it after
updating the code. `device="auto"` uses CUDA when available, otherwise CPU.

## 1. Import the library

Run this once after opening or restarting your notebook:

```python
from efficientgraphbench import Benchmark
```

## 2. Run your test

Start with this small local test:

```python
results = Benchmark(
    dataset="texas",
    models=["gcn", "mlp"],
    device="auto",
    epochs=20,
    seeds=[1, 2, 3],
    splits="official",
    output_dir="experiments/local-texas",
).run()
```

| Setting | What it means | What you can change |
|---|---|---|
| `dataset` | Data to test | For example, `"cora"` or `"pubmed"` |
| `models` | Models to compare | `"all"` or a list such as `["gcn", "gat"]` |
| `device` | Where training runs | `"cuda"` for GPU; `"auto"` to allow CPU fallback |
| `epochs` | Maximum training rounds per run | `200` for a benchmark; `2` for a quick check |
| `seeds` | Repeated runs with different random starts | `[1, 2, 3]` runs each model three times; `[42]` runs once |

`"all"` includes the primary models, with optional PyG SGFormer available by explicit ID. Official Graphormer will show `UNSUPPORTED_TASK` for this task; other models continue. GraphGPS and SGFormer require their separate environments to be installed.

## 3. See the results

Run this after Step 2 to see metrics, notes, recommendations and charts together:

```python
report = results.show_all()
```

- `runs`: number of successful runs.
- `accuracy_mean`: average accuracy; `0.81` means **81%**.
- `status`: `SUCCESS`, a failure reason category, or skipped runs.

The report is also saved at `experiments/local-texas/analysis/analysis.html`.
For just the formatted tables use `results.show()`; for just Pareto analysis use
`analysis = results.pareto()`. Separate plotting commands remain available.

To display saved results without running training again:

```python
from efficientgraphbench.results.dashboard import load_analysis_results

results = load_analysis_results("experiments/local-texas")
report = results.show_all()
```

Terminal equivalent (PowerShell on your PC):

```powershell
egbench analyze --input experiments/local-texas --open
```

On SSH omit `--open`, then copy the HTML report to your laptop to view the charts.

**That is the basic workflow: import → run → view.** The examples below are optional.

---

## Choose another task

| I want to… | Go to |
|---|---|
| Test only selected models | [Selected models](#selected-models) |
| Test one model | [One model](#one-model) |
| Test my CSV files | [My CSV files](#my-csv-files) |
| Increase graph size until OOM | [Size test](#size-test) |
| See speed and memory | [Speed and memory](#speed-and-memory) |
| Save my results | [Save results](#save-results) |
| Understand a failed run | [Check a failure](#check-a-failure) |

## Selected models

Use this **instead of Step 2**, then run Step 3:

```python
results = Benchmark(
    dataset="cora",
    models=["gcn", "graphsage", "sgformer", "graphgps"],
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()
```

## One model

```python
result = Benchmark.run_model(
    model="gcn",
    dataset="cora",
    device="cuda",
    epochs=200,
)

print("Status:", result.status)
print("Accuracy:", result.accuracy)
```

`result` is one run. `results` in the other examples contains multiple runs.

## My CSV files

Your prepared folder should contain `nodes.csv` and `edges.csv`:

```text
data/
    Mydata/
        nodes.csv
        edges.csv
```

```python
results = Benchmark(
    dataset="custom",
    dataset_path="data/Mydata",
    models="all",
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()
```

Then run Step 3. These must be prepared graph tables: nodes need `node_id`, `feature_*`, `label`, `split`; edges need `source`, `target`. An arbitrary spreadsheet needs preparation first.

Paths are relative to the notebook's working directory. To see that directory:

```python
from pathlib import Path
print(Path.cwd())
```

If necessary, set `dataset_path` to your folder's full path.

## Size test

Tests 2,000 → 4,000 → 6,000 nodes and so on, up to the full dataset:

```python
from efficientgraphbench import ScalingBenchmark

results = ScalingBenchmark(
    dataset="pubmed",
    models="all",
    step=2000,
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()

display(results.to_dataframe()[["model", "num_nodes", "accuracy", "status", "oom_nodes"]])
```

Shows **one final row per model**. After OOM, retains its last successful result and shows where OOM occurred. Remaining seeds and larger sizes for that model are skipped. Each size trains a fresh model on a subgraph; this is not mini-batch training.

## Speed and memory

After a comparison, run:

```python
display(results.summary()[[
    "model",
    "training_time_sec",
    "inference_latency_median_ms",
    "peak_gpu_allocated_mib",
]])
```

Columns show average training seconds, average per-run median inference milliseconds, and average peak allocated GPU memory in MiB.

## Save results

Choose the format you need; you do not need to run all three:

```python
results.save_csv("my_results.csv")        # Open as a spreadsheet
results.save_json("my_results.json")      # Complete records and settings
results.save_markdown("my_report.md")     # Readable report
```

Normal runs also save records, logs and checkpoints under `experiments/` by default. To choose a different folder, add `output_dir="experiments/my_test"` inside `Benchmark(...)`.

## Check a failure

For a comparison:

```python
for result in results:
    if result.status != "SUCCESS":
        print(result.model, result.status, result.failure_reason)
```

For a single-model run, use `print(result.failure_reason)`.

| Message | What to do |
|---|---|
| CUDA unavailable | Select the CUDA kernel, or use `device="auto"` |
| Environment not installed | Set up that reference model; see below |
| OOM | Use the size test to find a smaller working graph |
| Unsupported task | That implementation cannot run the current task |
| Skipped after a failure | Read the first failure; the repeat was not executed |

---

## Installation — only when needed

**Your current laptop is already set up. Skip this section there.**

For a new notebook environment, install the wheel in a notebook cell:

```python
%pip install "path/to/efficientgraphbench-0.1.0-py3-none-any.whl"
```

Restart the kernel afterward. For reference models, install Git on the computer, then run these commands in a **terminal**, not as Python code:

```bash
python -m pip install uv
egbench setup sgformer
egbench setup graphgps
```

Reference setup currently supports Windows x86-64.

## Full reference

Need advanced settings, provenance, JSONL exports, or terminal commands? Open [PYTHON_API_REFERENCE.md](PYTHON_API_REFERENCE.md). It contains the complete reference; this guide covers everyday notebook use.

Linux update: GraphGPS now has a Linux x86-64 installation path; see [GraphGPS on Linux](docs/graphgps_linux.md). It requires verification on the target server. SGFormer automatic setup remains Windows-only.
## Classification metrics and node-count plots

**Everything together:** `report = results.show_all()` displays metrics, notes,
all-model statistics, Pareto recommendations and available charts. See [Pareto and combined results](docs/pareto_and_analysis.md)
for the terminal command, reloading saved runs and optional constraints.

After running a benchmark, show its table, unusual results and the short macro/micro note:

```python
results.show()
```

This displays separate classification and performance tables in notebooks and
terminals. Scores use percentages, times show their units, missing values use a
dash, and unusual results are marked with `*` and explained below the tables.
`results.summary()` remains the numeric DataFrame for analysis and CSV exports.

The table includes Accuracy, support-weighted Precision/Recall/F1, Macro-F1 and Micro-F1.
Macro gives every evaluated class equal weight. Micro pools predictions; for these
single-label tasks its precision, recall and F1 equal accuracy. Notes require at least five distinct runs and a pattern in at least 80% of eligible runs. A persistent Macro-F1/Micro-F1 gap must be at least 15 percentage points with at least 100 test nodes per run. Zero recall requires at least 20 test examples of that class per run. These are descriptive screening rules, not significance tests; scores remain visible even when support is too small for a note.

```python
# All averages, including macro/micro precision and recall:
display(results.summary(detailed=True))
print(results.metric_notes())

# Plot every measured size from a scaling experiment:
from efficientgraphbench.results.plots import plot_training_time
plot_training_time("experiments/scaling/scaling-results.json", "experiments/scaling/plots")
```

Terminal equivalent:

```bash
egbench plot --runs experiments/scaling/scaling-results.json --output-dir experiments/scaling/plots
```

The line plot uses nodes on the x-axis and training seconds on the y-axis, averages
successful seeds at each size and excludes failed runs. Use the full
`scaling-results.json`, because the returned scaling table contains only the final
result for each model. Existing result files lack the new classification metrics;
rerun tests to calculate them. Metrics are not inferred from old accuracy values.


## Progressive training as the graph grows

```python
from efficientgraphbench import ProgressiveBenchmark

results = ProgressiveBenchmark(
    dataset="texas", models=["gcn", "graphsage", "mlp"], seeds=[1, 2, 3],
    device="auto", output_dir="experiments/progressive-texas",
    progressive={
        "progression_seed": 42, "selection_metric": "val_f1_macro", "min_candidates": 2,
        "stages": [
            {"name": "initial", "graph_fraction": 0.5, "additional_epochs": 10,
             "retain_fraction": 0.7},
            {"name": "full", "graph_fraction": 1.0, "additional_epochs": 20},
        ],
    },
).run()
results.show_all()
```

Each surviving model resumes its own weights and optimizer at the next size. See [the progressive guide](docs/progressive_benchmarking.md) for stage settings, capability limits and saved evidence. Ordinary benchmarks remain unchanged. Official SGFormer and GraphGPS now support progressive checkpoint continuation through their installed reference environments. Graphormer remains unsupported; `sgformer_pyg` remains a separate identity.


### Progressive GNNs and official transformers

```python
from efficientgraphbench import ProgressiveBenchmark

results = ProgressiveBenchmark(
    "texas", models=["gcn", "graphsage", "sgformer", "graphgps"],
    seeds=[1, 2, 3], device="auto",
    output_dir="experiments/progressive-transformers",
    progressive={
        "progression_strategy": "deterministic_nested",
        "selection_metric": "val_f1_macro",
        "min_candidates": 2,
        "stages": [
            {"name": "initial", "graph_fraction": 0.5, "additional_epochs": 10, "retain_fraction": 0.7},
            {"name": "full", "graph_fraction": 1.0, "additional_epochs": 20},
        ],
    },
).run()
results.show_all()
```

Official transformers require `egbench setup sgformer` and `egbench setup graphgps` first. Existing platform restrictions still apply; automatic SGFormer setup currently targets Windows. To print the report, open the returned HTML in a browser, choose Print, enable background graphics, and save as PDF in portrait orientation.

For a measured comparison, explicitly run Full separately with the same models/seeds and `epochs=30`; then set `full_baseline` inside `progressive` to that Full experiment directory. This compares different learning protocols; it does not make their training exposure identical.

For the college Ubuntu GPU, follow [the SSH update and Progressive run instructions](docs/college_progressive.md). Ready-to-run configurations are `configs/college_progressive_smoke.yaml` (short Texas CUDA check), `configs/college_progressive_pubmed.yaml` (10 seeds), and `configs/college_progressive_graphgps.yaml` (official GraphGPS Linux verification). They run with `egbench progressive --config <file>`.
