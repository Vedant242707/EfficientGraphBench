# EfficientGraphBench Python commands — v0.1.0

## Installation and help

Run terminal commands from the project directory for local installation:

```powershell
python -m pip install -e .
python -m pip install -e ".[dev,tables]"
python -m pip install uv
python -m efficientgraphbench --help
egbench --help
egbench run --help
```

After installing, `egbench` works from any directory. `python -m efficientgraphbench` accepts the same commands. In this checkout you may use `.venv\Scripts\python.exe` and `.venv\Scripts\egbench.exe` instead. The package is not yet published to PyPI.

## Import and version

```python
import efficientgraphbench as egb
from efficientgraphbench import Benchmark, ScalingBenchmark
from efficientgraphbench import BenchmarkResult, BenchmarkResults, HardwareProfile
print(egb.__version__)
```

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

```python
bench = Benchmark(
    dataset="cora", models=["gcn", "graphsage", "sgformer", "graphgps"],
    seeds=[1, 2, 3], device="cuda", epochs=200,
    output_dir="experiments/python-matrix",
)
results = bench.run()
print(results.summary())
```

`Benchmark(dataset="cora", models="all", seeds=(42,43,44), **options)` runs models sequentially. `models="all"` selects the primary models. An explicit list selects any registered implementations, including experimental variants. No subprocess/environment activation details are required from Python.

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

Syntax: `load_dataset(name, root="data", *, nodes=None, edges=None, dataset_path=None, split_seed=0, split_index=0)`. Supply either the pair of CSV paths or dataset_path. Named CSV files contain their own splits and do not accept split options. split_index applies only to WikiCS.

CSV nodes: node_id, feature_* columns, label, split. CSV edges: source, target, optional numeric attributes. `InvalidDatasetError` explains invalid columns, endpoints, features or splits. Each model's feature/edge support is still model-specific.

## Prepare and view original tables

Use the documented CLI `prepare`, `table`, `view` and `inspect` commands below. These preserve the existing import workflow; a PDF or spreadsheet must first contain explicit node/edge tables. The library does not infer relationships from arbitrary documents.

## Graph-size experiments

```python
experiment = ScalingBenchmark(
    model="gcn", dataset="pubmed", sizes=[2000,4000,8000,12000],
    seeds=[42,43,44], device="cuda", epochs=200,
    output_dir="experiments/python-scaling",
)
results = experiment.run()
print(results.summary())
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

Setup requires Git and uv, may download packages and creates an isolated environment. Current locks support Windows x86-64. verify=True checks the clean pinned checkout and model imports; it does not claim successful training on all graphs. `EGBENCH_HOME` overrides the cache location; set it before invoking the library. No manual activation is needed.

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


### Common configuration

| Argument | Default |
|---|---|
| `dataset` | `'cora'` |
| `device` | `'auto'` |
| `seed` | `42` |
| `threads` | `4` |
| `data_dir` | `'data'` |
| `output_dir` | `'experiments'` |
| `dataset_path` | `None` |
| `split_seed` | `0` |
| `split_index` | `0` |
| `model_options` | `{}` |
| `worker_timeout_sec` | `7200` |

### Training configuration

| Argument | Default |
|---|---|
| `epochs` | `200` |
| `lr` | `0.01` |
| `weight_decay` | `0.0005` |
| `latency_warmup` | `10` |
| `latency_repeats` | `50` |
| `optimizer` | `'Adam'` |
| `scheduler` | `None` |
| `early_stopping_patience` | `None` |

### Model configuration (via model_options where supported)

| Argument | Default |
|---|---|
| `name` | `'gcn'` |
| `hidden_dim` | `64` |
| `num_layers` | `2` |
| `dropout` | `0.5` |
| `heads` | `4` |
| `propagation_steps` | `10` |
| `alpha` | `0.1` |


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


### compare

Show the latest comparison table for this dataset.

`egbench compare --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'cora'` | dataset |
| `--output-dir` | `'experiments'` | output dir |
| `--report/--no-report` | `False` | report |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### datasets

List built-in datasets and the custom-file option. Does not download data.

`egbench datasets --help`

| Argument / option | Default | Use |
|---|---|---|
| `--help` | — | Show syntax and options; do not run a benchmark. |

### environments

Show installed reference environments and pinned commits.

`egbench environments --help`

| Argument / option | Default | Use |
|---|---|---|
| `--help` | — | Show syntax and options; do not run a benchmark. |

### hardware

Show whether this environment can use your GPU.

`egbench hardware --help`

| Argument / option | Default | Use |
|---|---|---|
| `--help` | — | Show syntax and options; do not run a benchmark. |

### inspect

Load and check data without training. Built-in datasets download on first use.

`egbench inspect --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `None` | dataset |
| `--dataset-path` | `None` | dataset path |
| `--data-dir` | `None` | data dir |
| `--split-seed` | `None` | split seed |
| `--split-index` | `None` | split index |
| `--config` | `None` | config |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### matrix

Run primary models sequentially; --models selects comma-separated model IDs.

`egbench matrix --help`

| Argument / option | Default | Use |
|---|---|---|
| `--config` | `None` | config |
| `--dataset` | `None` | dataset |
| `--seeds` | `'42,43,44'` | seeds |
| `--epochs` | `None` | epochs |
| `--device` | `None` | device |
| `--output-dir` | `None` | output dir |
| `--dataset-path` | `None` | dataset path |
| `--split-seed` | `None` | split seed |
| `--split-index` | `None` | split index |
| `--models` | `None` | models |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### models

List model sources, environment readiness, and experimental alternatives.

`egbench models --help`

| Argument / option | Default | Use |
|---|---|---|
| `name` | `None` | name |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### prepare

Map CSV, TSV, XLSX, or PDF tables into a validated graph dataset.

`egbench prepare --help`

| Argument / option | Default | Use |
|---|---|---|
| `--nodes/-n` | `required` | nodes |
| `--edges/-e` | `required` | edges |
| `--output-dir` | `'data/my_dataset'` | output dir |
| `--node-id` | `None` | node id |
| `--label` | `None` | label |
| `--source` | `None` | source |
| `--target` | `None` | target |
| `--features` | `None` | features |
| `--split-column` | `'split'` | split column |
| `--split-seed` | `0` | split seed |
| `--nodes-sheet` | `None` | nodes sheet |
| `--edges-sheet` | `None` | edges sheet |
| `--nodes-page` | `None` | nodes page |
| `--edges-page` | `None` | edges page |
| `--nodes-table` | `None` | nodes table |
| `--edges-table` | `None` | edges table |
| `--nodes-header-row` | `1` | nodes header row |
| `--edges-header-row` | `1` | edges header row |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### preview

Show columns and the first five rows before mapping a table into a graph.

`egbench preview --help`

| Argument / option | Default | Use |
|---|---|---|
| `--file` | `required` | file |
| `--sheet` | `None` | sheet |
| `--page` | `None` | page |
| `--table` | `None` | table |
| `--header-row` | `1` | header row |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### provenance

Write model sources, versions, modifications and measured parameter counts.

`egbench provenance --help`

| Argument / option | Default | Use |
|---|---|---|
| `--output` | `'audits/review20/model_provenance.json'` | output |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### recommend

Filter primary measured results; GPU budget uses maximum reserved MiB across seeds.

`egbench recommend --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'cora'` | dataset |
| `--output-dir` | `'experiments'` | output dir |
| `--minimum-accuracy` | `0.8` | minimum accuracy |
| `--maximum-latency-ms` | `10.0` | maximum latency ms |
| `--maximum-gpu-mib` | `6141.0` | maximum gpu mib |
| `--group` | `None` | group |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### report

Generate a Markdown report from saved benchmark results.

`egbench report --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'cora'` | dataset |
| `--output-dir` | `'experiments'` | output dir |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### run

Train one model; explicit CLI options override YAML settings.

`egbench run --help`

| Argument / option | Default | Use |
|---|---|---|
| `--config` | `None` | config |
| `--dataset` | `None` | dataset |
| `--model` | `None` | model |
| `--seed` | `None` | seed |
| `--epochs` | `None` | epochs |
| `--device` | `None` | device |
| `--output-dir` | `None` | output dir |
| `--dataset-path` | `None` | dataset path |
| `--split-seed` | `None` | split seed |
| `--split-index` | `None` | split index |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### scale

Measure deterministic induced subgraphs; retains original split memberships.

`egbench scale --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'cora'` | dataset |
| `--sizes` | `'500,1000,2708'` | sizes |
| `--models/--model` | `'gcn,graphsage,gat,gatv2,appnp,sgc,mlp,sgformer,graphgps,graphormer'` | models |
| `--seeds` | `'42,43,44'` | seeds |
| `--epochs` | `20` | epochs |
| `--device` | `'auto'` | device |
| `--output-dir` | `'experiments/scaling'` | output dir |
| `--dataset-path` | `None` | dataset path |
| `--step` | `None` | Increase node count by this amount up to the full dataset; overrides --sizes. |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### scaling

Measure deterministic induced subgraphs; retains original split memberships.

`egbench scaling --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'cora'` | dataset |
| `--sizes` | `'500,1000,2708'` | sizes |
| `--models/--model` | `'gcn,graphsage,gat,gatv2,appnp,sgc,mlp,sgformer,graphgps,graphormer'` | models |
| `--seeds` | `'42,43,44'` | seeds |
| `--epochs` | `20` | epochs |
| `--device` | `'auto'` | device |
| `--output-dir` | `'experiments/scaling'` | output dir |
| `--dataset-path` | `None` | dataset path |
| `--step` | `None` | Increase node count by this amount up to the full dataset; overrides --sizes. |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### setup

Install and verify an isolated reference environment.

`egbench setup --help`

| Argument / option | Default | Use |
|---|---|---|
| `model` | `required` | model |
| `--help` | — | Show syntax and options; do not run a benchmark. |

### view

Create a searchable HTML data sheet; open the saved file in your browser.

`egbench view --help`

| Argument / option | Default | Use |
|---|---|---|
| `--dataset` | `'Cora'` | dataset |
| `--dataset-path` | `None` | dataset path |
| `--file` | `None` | file |
| `--sheet` | `None` | sheet |
| `--page` | `None` | page |
| `--table` | `None` | table |
| `--header-row` | `1` | header row |
| `--data-dir` | `'data'` | data dir |
| `--columns` | `None` | columns |
| `--output` | `'data/dataset-view.html'` | output |
| `--help` | — | Show syntax and options; do not run a benchmark. |

## Development commands

```bash
python -m pytest -q
python -m build
```
