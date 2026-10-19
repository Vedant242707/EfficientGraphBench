# ruff: noqa: E501
"""Generate the public Python and CLI command reference from installed interfaces."""

from dataclasses import fields
from pathlib import Path

import typer.main

from efficientgraphbench.cli.app import app
from efficientgraphbench.config import BenchmarkConfig, ModelConfig, TrainingConfig

INTRO = """# EfficientGraphBench Python commands — v0.1.0

## Installation and help

Updated 4 October 2026. Run Python blocks in your notebook/script and terminal blocks
in PowerShell or your activated SSH shell. Run examples individually, not all at once.
After a code update, restart the notebook kernel and select the project environment.

Run terminal commands from the project directory for local installation:

```powershell
python -m pip install -e .
python -m pip install -e ".[dev,tables]"
python -m pip install -e ".[ogb,plots]"
python -m pip install uv
python -m efficientgraphbench --help
egbench --help
egbench run --help
```

After installing, `egbench` works from any directory. `python -m efficientgraphbench` accepts the same commands. In this checkout you may use `.venv\\Scripts\\python.exe` and `.venv\\Scripts\\egbench.exe` instead. The package is not yet published to PyPI.

## Import and version

```python
import efficientgraphbench as egb
from efficientgraphbench import Benchmark, ScalingBenchmark
from efficientgraphbench import BenchmarkResult, BenchmarkResults, HardwareProfile
print(egb.__version__)
```

## Quick local test and everything together

```python
results = Benchmark(
    dataset="texas", models=["gcn", "mlp"], runs=3,
    splits="official", epochs=20, device="auto",
    output_dir="experiments/local-texas",
).run()
report = results.show_all()
```

`show_all()` displays classification metrics, performance, unusual-result notes,
all-model statistics, Pareto recommendations and available plots together in Jupyter. It saves a shareable
HTML report with embedded images, JSON analysis, summary CSV and PNG/PDF plots.
No retraining occurs when displaying or analyzing results. CPU runs do not have GPU
allocation charts; a single node count produces one point on the training-time plot.

```python
from efficientgraphbench.results.dashboard import load_analysis_results
results = load_analysis_results("experiments/local-texas")
report = results.show_all()  # Reload and display saved tests without training
report = results.show_all(output_dir="exports/analysis", display=False)
report = results.show_all(open_browser=True)
```

`load_analysis_results(path)` accepts an experiment directory, run JSON/JSONL or
`scaling-results.json`. For scaling it retains all seeds at each model's final
successful size for the tables and all measured sizes for the line plot.

## Classification metrics and formatted output

```python
results.show()                         # Classification/performance tables and notes
table = results.summary()             # Numeric DataFrame, not the formatted display
table = results.summary(detailed=True) # Also macro/micro precision and recall
print(results.metric_notes())         # Unusual results and averaging explanation
```

Always shown: Accuracy, support-weighted Precision/Recall/F1, Macro-F1 and Micro-F1.
Displayed scores are percentages; raw values are fractions. Macro treats evaluated
classes equally; micro pools predictions and equals accuracy for these single-label
tasks. Notes require at least five distinct runs and a pattern in at least 80% of eligible runs. A persistent Macro-F1/Micro-F1 gap must be at least 15 percentage points with at least 100 test nodes per run. Zero recall requires at least 20 test examples of that class per run. These are descriptive screening rules, not significance tests; scores remain visible even when support is too small for a note. Old runs cannot acquire these metrics from
accuracy alone; rerun tests if the values are missing.

## Pareto frontier and multi-objective recommendations

```python
analysis = results.pareto()
analysis = results.pareto(
    minimum_quality=0.70, minimum_accuracy=0.75,
    maximum_latency_ms=20, minimum_runs=10,
    include_training=True,
)
# The same options are accepted by results.show_all(...).
```

Thresholds above are examples, not universal requirements. CUDA defaults maximize
Macro-F1 and minimize inference latency and maximum peak reserved GPU memory.
CPU auto mode uses quality and latency. Matched seed/split runs must be complete;
different graphs, hardware and budgets are analyzed separately. Qualified means
meeting limits; frontier means no other qualified candidate is at least as good on
every selected objective and strictly better on one. Ties stay. This is descriptive
analysis, not proof of statistical superiority.

| Argument | Default | Use |
|---|---|---|
| `quality` | `"f1_macro"` | Choose `"f1_macro"` or `"accuracy"` as quality objective. |
| `memory` | `"auto"` | `"gpu"`, `"cpu"`, `"none"`, or automatic CUDA memory selection. CPU RSS is process memory, not isolated model memory. |
| `include_training` | `False` | Include mean training seconds in Pareto dominance. |
| `minimum_quality` | `0.0` | Minimum selected quality score, as a fraction. |
| `minimum_accuracy` | `0.0` | Minimum mean accuracy, as a fraction. |
| `maximum_latency_ms` | `None` | Maximum mean per-run median latency. |
| `maximum_memory_mib` | `None` | Maximum run peak of the selected memory metric. |
| `minimum_runs` | `1` | Minimum matched successful runs per candidate. |
| `weights` | `None` | Optional explicit balanced-selection weights, Python only. |

Recommendations include best quality, fastest inference, fastest training, lowest
memory when selected, and the overall frontier shortlist. Optional weights must
cover every active objective with nonnegative values and positive sum. For CUDA:
`weights={"quality": 0.6, "latency_ms": 0.3, "memory_mib": 0.1}`. Add
`"training_time_sec"` when including training; omit `"memory_mib"` for CPU auto.
Balanced selection minimizes weighted normalized losses among frontier candidates.

## Dataset profiling

```python
from efficientgraphbench.datasets import profile_dataset
profile = profile_dataset("texas")
print(profile)
metrics = profile.to_dict()
profile.save_json("exports/texas-profile.json")
```

`profile_dataset(dataset, root="data", **load_options)` accepts a dataset name or
loaded DatasetBundle. It reports nodes, edges, features, classes, density, degree
statistics, homophily, class imbalance, components and feature sparsity. It does
not train models. `data.profile()` does the same for a loaded dataset.

## Plot saved results separately

```python
from efficientgraphbench.results.plots import plot_summary, plot_training_time
plot_summary(results.summary(), "experiments/local-texas/plots")
plot_training_time(
    "experiments/scaling/scaling-results.json",
    "experiments/scaling/plots",
)
```

`plot_summary(summary, output_dir)` accepts a DataFrame or summary CSV and exports
accuracy/latency and GPU allocation charts, plus node-count/training-time when
available. `plot_training_time(records, output_dir)` accepts raw records,
BenchmarkResults, DataFrame, or JSON/JSONL/CSV. It averages successful runs at each
node count with sample-SD bars and keeps different recorded protocols separate.
Both return exported image/PDF paths. Use complete scaling records for a curve.

## Run one model

```python
result = Benchmark.run_model(
    model="gcn", dataset="cora", device="cuda", seed=42,
    epochs=200, output_dir="experiments/python-single",
)
print(result.status)
print(result.accuracy)
print(result.training_time)
print(result.preprocessing_time)
print(result.inference_latency)
print(result.gpu_memory)
print(result.system_memory)
print(result.failure_reason)
record = result.to_dict()
```

`run_model(model, dataset="cora", **options)` trains one model and returns BenchmarkResult. `dataset` can be a built-in ID or a DatasetBundle. Failed runs are results; invalid Python arguments or dataset objects raise descriptive exceptions. Missing metrics are None. Time is seconds, latency milliseconds, memory MiB. `gpu_memory` is peak allocated memory, `system_memory` is sampled process RSS.

Additional result properties: `model`, `model_source`, `repository_commit`, `dataset`, `dataset_fingerprint`, `split_hash`, `seed`, `hardware`, `software`, `validation_accuracy`, `latency_mean`, `latency_median`, `latency_std`, `peak_gpu_allocated`, `peak_gpu_reserved`, `peak_system_ram`, `parameter_count`, `trainable_parameter_count`, `epochs`, `failure_stage`. Canonical JSON fields are also accessible by name or through `result.record`.

## Run a model comparison

For 10–20 runs use `Benchmark(dataset="ogbn-arxiv", models=["gcn"], runs=10, device="cuda").run()`. `runs` generates seeds 1..N and cannot be combined with `seeds`. `splits="fixed"` preserves a single split (default), `splits="official"` cycles available official folds, and `splits="random"` generates stratified 60/20/20 splits per seed. Random OGB splits are nonofficial. Repeated split summaries retain hardware/configuration boundaries and include attempted runs and distinct successful splits. `summary(across_splits=False)` separates folds; `summary(across_splits=True)` explicitly aggregates compatible folds from reloaded records. See [College experiments](docs/college_experiments.md) for Python, SSH notebooks, datasets, and detailed timing/memory fields. The CLI equivalent is `egbench repeat --dataset ogbn-arxiv --runs 10`; use `egbench repeat --help` for options.

```python
bench = Benchmark(
    dataset="cora", models=["gcn", "graphsage", "sgformer", "graphgps"],
    seeds=[1, 2, 3], device="cuda", epochs=200,
    output_dir="experiments/python-matrix",
)
results = bench.run()
results.show()
```

`Benchmark(dataset="cora", models="all", seeds=(42,43,44), **options)` runs models sequentially. `models="all"` selects the primary models. An explicit list selects any registered implementations, including the maintained PyG SGFormer alternative. No subprocess/environment activation details are required from Python.

After OOM, remaining seeds are skipped. Pretraining deterministic failures also skip remaining seeds. Skipped rows have no measured accuracy/timing. `SKIPPED_DETERMINISTIC_FAILURE` and `SKIPPED_AFTER_OOM` distinguish these policies. Other models continue.

## Load datasets

```python
from efficientgraphbench.datasets import load_dataset, DatasetBundle, InvalidDatasetError

data = load_dataset("cora", root="data")
data = load_dataset("custom", nodes="data/Mydata/nodes.csv", edges="data/Mydata/edges.csv")
data = load_dataset("custom", dataset_path="data/Mydata")
data = load_dataset("custom", dataset_path="data/Mydata/graph.npz")
print(data.metadata())
print(data.data.x)
print(data.data.edge_index)
print(data.data.y)
print(data.data.train_mask)
result = Benchmark.run_model("gcn", dataset=data, epochs=200)
```

Syntax: `load_dataset(name, root="data", *, nodes=None, edges=None, **options)`. Options include dataset_path, split_seed, split_index and split_mode. Supply either the CSV pair or dataset_path. Named CSV files contain splits and do not accept extra options. split_index selects an available official fold (WikiCS or supported heterophilous datasets).

CSV nodes: node_id, feature_* columns, label, split. CSV edges: source, target, optional numeric attributes. `InvalidDatasetError` explains invalid columns, endpoints, features or splits. Each model's feature/edge support is still model-specific.

## Prepare and view original tables

Use the CLI `prepare`, `preview`, `view` and `inspect` commands below. A PDF or spreadsheet must contain explicit node/edge tables; relationships are not inferred from arbitrary documents.

## Graph-size experiments

```python
experiment = ScalingBenchmark(
    model="gcn", dataset="pubmed", sizes=[2000,4000,8000,12000],
    seeds=[42,43,44], device="cuda", epochs=200,
    output_dir="experiments/python-scaling",
)
results = experiment.run()
report = results.show_all()
print(results.to_dataframe()[["model", "num_nodes", "status", "oom_nodes"]])
print(results.all_results_path)
```

Syntax: `ScalingBenchmark(dataset="cora", model=None, models=None, sizes=None, step=None, seeds=(42,43,44), **options)`. Choose model or models, not both. `step=2000` replaces sizes and includes the full graph as the final size. Sizes count nodes, not tokens. Each size is a fresh induced-subgraph experiment, not full-graph mini-batching. After OOM, skip remaining seeds/larger sizes. Returned results contain only the last executed result per model, retaining the last successful result if the next attempt OOMs; all measured sizes remain in the JSON file. No OOM means only the final tested size is returned.

## Results and exports

```python
print(len(results))
first = results[0]
for result in results:
    print(result.model, result.status)
frame = results.to_dataframe()      # One row per result, including failures/skips
summary = results.summary()         # Aggregate successful seeds within comparison groups
results.save_csv("exports/results.csv")
results.save_json("exports/results.json")
results.save_jsonl("exports/results.jsonl")
results.save_markdown("exports/report.md")
```

Construct `BenchmarkResults(records, output_dir=None)` from a list of record dictionaries to reload saved results. Export methods return their output Path. JSON retains complete nested provenance; CSV serializes nested structures as JSON cells. Summary never combines different graph/hardware/split comparison groups. Failures are retained; missing metrics are not replaced with zero.

```python
import json
saved = BenchmarkResults(json.loads(open("exports/results.json", encoding="utf-8").read()))
```

## Hardware

```python
hardware = HardwareProfile.collect()
print(hardware.to_dict())
print(hardware.cpu)
print(hardware.ram_bytes)
print(hardware.device_name)
print(hardware.cuda_available)
print(hardware.cuda_version)
```

Collects actual machine information. RAM capacity is bytes; VRAM fields ending in _mib or _mb are MiB. GPU values may be None on CPU-only systems.

## Inspect models

```python
from efficientgraphbench.models.metadata import get_model, list_models
print(get_model("graphormer"))
for model in list_models():
    print(model["display_name"], model["status"])
```

Model metadata includes source, runner type, environment, supported tasks, repository, paper, category and readiness. Official Graphormer remains UNSUPPORTED_TASK for current node classification.

## Set up isolated environments

```python
from efficientgraphbench.environments import setup_environment, inspect_environment
setup_environment("graphgps")
setup_environment("sgformer")
print(inspect_environment("graphgps"))
print(inspect_environment("graphgps", verify=True))
```

Setup requires Git and uv, may download packages and creates an isolated environment. Current locks support Windows x86-64; GraphGPS also has a [Linux x86-64 setup path](docs/graphgps_linux.md), which must be verified on the target server. SGFormer automatic setup remains Windows-only. verify=True checks the clean pinned checkout and model imports; it does not claim successful training on all graphs. `EGBENCH_HOME` overrides the cache location; set it before invoking the library. No manual activation is needed.

## Reports, provenance and recommendation

```python
from efficientgraphbench.results.report import write_report
from efficientgraphbench.results.writer import read_results
from efficientgraphbench.models.provenance import generate_provenance
from efficientgraphbench.results.recommend import recommend

write_report("cora", "experiments/python-matrix")
generate_provenance("exports/model_provenance.json")
records = read_results("experiments/python-matrix")
choices = recommend(records, minimum_accuracy=0.8, maximum_latency_ms=10,
                    maximum_gpu_mib=6000, group=None)
```

Recommendations use comparable successful primary measurements, require all expected seeds, and use peak reserved GPU memory. Specify a comparison group if multiple groups are present. No recommendation is fabricated for unmeasured models.

## Benchmark options

`Benchmark`, `Benchmark.run_model` and `ScalingBenchmark` accept the following common keyword options. `model` is selected by the public method, and dataset is a name or DatasetBundle. `training` optionally accepts TrainingConfig or a dictionary; training fields may also be passed directly (`epochs=200`). `model_options` contains model-specific overrides validated by the existing engine.
"""

out = [
    INTRO,
    '\n## Progressive graph growth\n\nUse `ProgressiveBenchmark` or `Benchmark(..., benchmark_mode="progressive_multi_fidelity", progressive={...})`. Survivors resume weights, optimizer and scheduler across nested graph stages. See [the progressive guide](docs/progressive_benchmarking.md) for complete Python examples, configurations, capability limits and reports. `results.show_all()` shows the progressive report; `results.summary()` returns its DataFrame.\n',
]
for title, cls in [
    ("Common configuration", BenchmarkConfig),
    ("Training configuration", TrainingConfig),
    ("Model configuration (via model_options where supported)", ModelConfig),
]:
    out += ["\n### " + title, "", "| Argument | Default |", "|---|---|"]
    instance = cls()
    for field in fields(cls):
        if field.name in {"model", "training"}:
            continue
        out.append(f"| `{field.name}` | `{getattr(instance, field.name)!r}` |")
out += [
    """

Reference GraphGPS accepts model_options: lr, weight_decay, layers, hidden_dim, heads, dropout.
Reference SGFormer additionally accepts trans_layers, trans_dropout, graph_weight, alpha,
trans_weight_decay, use_bn, use_residual, use_weight, use_act, patience.
In-process models accept hidden_dim, num_layers, dropout, heads, propagation_steps, alpha
and the training override keys lr, weight_decay, optimizer, scheduler, early_stopping_patience.
Not every setting affects every architecture. Graphormer reference remains task-gated.

## All terminal commands and options

Every command below can be used as `egbench COMMAND` or `python -m efficientgraphbench COMMAND`.
Every command supports `--help`. Options are generated from the actual CLI to avoid stale names.
`--seeds 3` means seed ID 3; use `--seeds 1,2,3` for three runs.
"""
]
EXAMPLES = {
    "progressive": "egbench progressive --config configs/progressive_texas.yaml",
    "analyze": "egbench analyze --input experiments/local-texas --open",
    "pareto": "egbench pareto --input experiments/local-texas",
    "datasets": "egbench datasets",
    "models": "egbench models gcn",
    "hardware": "egbench hardware",
    "environments": "egbench environments",
    "setup": "egbench setup graphgps",
    "profile": "egbench profile --dataset texas --output exports/texas-profile.json",
    "inspect": "egbench inspect --dataset texas",
    "preview": "egbench preview --file data/Mydata/nodes.csv",
    "view": "egbench view --dataset texas --output data/texas-view.html",
    "prepare": "egbench prepare --nodes data/Mydata/nodes.csv --edges data/Mydata/edges.csv --output-dir data/prepared",
    "run": "egbench run --dataset texas --model gcn --epochs 20 --device auto",
    "matrix": "egbench matrix --dataset texas --models gcn,mlp --seeds 1,2,3 --epochs 20 --device auto",
    "repeat": "egbench repeat --dataset texas --models gcn,mlp --runs 3 --splits official --epochs 20 --device auto --output-dir experiments/local-texas",
    "scale": "egbench scale --dataset texas --models gcn,mlp --sizes 60,120,183 --epochs 20 --device auto",
    "scaling": "egbench scaling --dataset texas --models gcn --sizes 60,120,183 --epochs 20 --device auto",
    "compare": "egbench compare --dataset texas --output-dir experiments/local-texas",
    "report": "egbench report --dataset texas --output-dir experiments/local-texas",
    "plot": "egbench plot --runs experiments/scaling/scaling-results.json --output-dir experiments/scaling/plots",
    "provenance": "egbench provenance --output exports/model_provenance.json",
    "recommend": "egbench recommend --dataset texas --output-dir experiments/local-texas",
}
OPTION_USES = {
    "dataset": "Registered dataset ID to load or select.",
    "dataset_path": "Prepared custom CSV folder or numeric NPZ graph path.",
    "data_dir": "Root directory for downloaded/cached datasets.",
    "model": "Model identifier; setup accepts a reference environment name.",
    "models": "Comma-separated model identifiers.",
    "seed": "One training random seed.",
    "seeds": "Comma-separated seed IDs; 1,2,3 means three runs.",
    "epochs": "Maximum training epochs per run.",
    "device": "auto, cpu, or cuda; CUDA requires a working GPU in this Python host.",
    "runs": "Number of runs (repeat), or saved run file (plot).",
    "splits": "fixed, official (cycle folds), or random (60/20/20 per seed).",
    "split_seed": "Seed for generated split or size-sampling order.",
    "split_index": "Index of an available official fold.",
    "output_dir": "Directory for outputs, or existing results for compare/report/recommend.",
    "output": "Destination file path.",
    "input": "Existing experiment directory or saved run JSON/JSONL.",
    "summary": "Summary CSV for plots; use either this or --runs.",
    "config": "YAML configuration; explicit command options override it.",
    "file": "Input CSV, TSV, XLSX or table-based PDF.",
    "nodes": "Input node table with IDs, numeric features and labels.",
    "edges": "Input edge table with source and target IDs.",
    "node_id": "Column containing unique node identifiers.",
    "label": "Column containing the prediction target/class.",
    "source": "Edge-table column containing source node IDs.",
    "target": "Edge-table column containing target node IDs.",
    "features": "Comma-separated numeric node-feature columns.",
    "columns": "Comma-separated columns to show in the data sheet.",
    "split_column": "Column with train/val/test assignments, if present.",
    "sizes": "Comma-separated induced-subgraph node counts.",
    "step": "Increase nodes by this amount; overrides --sizes.",
    "quality": "Pareto quality objective: f1_macro or accuracy.",
    "memory": "Pareto memory objective: auto, gpu, cpu, or none.",
    "include_training": "Include training seconds as a Pareto objective.",
    "minimum_quality": "Minimum mean quality as a fraction from 0 to 1.",
    "minimum_accuracy": "Minimum mean accuracy as a fraction from 0 to 1.",
    "maximum_latency_ms": "Maximum mean per-run median inference latency in milliseconds.",
    "maximum_memory_mib": "Maximum run peak of the selected memory metric, in MiB.",
    "maximum_gpu_mib": "Maximum run peak reserved GPU memory, in MiB.",
    "minimum_runs": "Minimum complete matched seed/split runs for recommendation.",
    "group": "Explicit comparison-group ID when multiple groups exist.",
    "open_browser": "Open the saved combined HTML report in the local browser.",
    "report": "Also write a Markdown report.",
}
command = typer.main.get_command(app)
out += [
    "\n### Top-level help and shell completion\n",
    "`egbench --help`: list all commands. `egbench COMMAND --help`: list that command's options.",
    "`egbench --install-completion`: install completion for the current shell.",
    "`egbench --show-completion`: print the shell completion script.",
]
for name, cmd in sorted(command.commands.items()):
    out += [
        "\n### " + name,
        "",
        (cmd.help or "").strip(),
        "",
        f"`egbench {name} --help`",
        "",
        "Example:",
        "```bash",
        EXAMPLES[name],
        "```",
        "",
        "| Argument / option | Default | Use |",
        "|---|---|---|",
    ]
    for param in cmd.params:
        options = "/".join(param.opts + getattr(param, "secondary_opts", []))
        default = "required" if param.required else repr(param.default)
        helptext = getattr(param, "help", "") or OPTION_USES.get(param.name, "")
        if not helptext:
            if param.name.endswith("sheet"):
                helptext = "Excel worksheet name to read; required when selection is ambiguous."
            elif param.name.endswith("header_row"):
                helptext = "One-based row containing column names."
            elif param.name.endswith("page"):
                helptext = "One-based PDF page number."
            elif param.name.endswith("table"):
                helptext = "One-based PDF table number on the selected page."
            else:
                helptext = param.name.replace("_", " ")
        helptext = helptext.replace("|", "\\|")
        out.append(f"| `{options}` | `{default}` | {helptext} |")
    out.append("| `--help` | — | Show syntax and options; do not run a benchmark. |")
out += ["\n## Development commands\n\n```bash\npython -m pytest -q\npython -m build\n```\n"]
Path("PYTHON_API_REFERENCE.md").write_text("\n".join(out), encoding="utf-8")
