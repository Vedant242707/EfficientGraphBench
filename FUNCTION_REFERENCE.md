Python library and installed CLI reference: [PYTHON_COMMANDS.md](PYTHON_COMMANDS.md).

# Complete command reference

## Reference benchmark commands

| Syntax | Use |
|---|---|
| `egbench models` | List implementation sources and environment readiness. |
| `egbench provenance [--output PATH]` | Generate current model provenance JSON. |
| `egbench matrix --models gcn,graphgps,sgformer` | Select models; default is the primary catalog. |
| `egbench scale --dataset cora --sizes 500,1000,2708 --models gcn,graphgps --seeds 42,43,44 --epochs 20 --device cuda --output-dir experiments/scaling` | Measure deterministic induced subgraphs with retained split membership. |
| `egbench scale --dataset custom --dataset-path data/my_dataset --sizes 100,200` | Run size experiments on a custom graph. Sizes cannot exceed the supplied graph. |
| `egbench recommend --dataset cora --output-dir experiments/reference-final --minimum-accuracy 0.8 --maximum-latency-ms 10 --maximum-gpu-mib 6141 [--group HASH]` | Filter complete primary measured results within one comparison group. |
| `egbench models --help` | Show model-list command help. |
| `egbench provenance --help` | Show provenance options. |
| `egbench scale --help` | Show graph-size experiment options. |
| `egbench recommend --help` | Show measured recommendation constraints. |

Reference IDs: `graphgps`, `sgformer`, `sgformer_reference`.
Library variant: `sgformer_pyg`. Experimental IDs: `graphgps_adapted`, `graphormer_adapted`,
`appnp_adapted`. Primary `appnp` uses PyG's pinned citation benchmark class.
`graphormer` returns `UNSUPPORTED_TASK`; its experimental alternative retains the custom4096 guard.
Reference recipe overrides use YAML `model_options`; `worker_timeout_sec` controls the external
process timeout. See [REFERENCE_BENCHMARKS.md](REFERENCE_BENCHMARKS.md) for supported model options.

## Start here

Run commands in PowerShell or CMD from the project folder. No virtual-environment activation is needed when using the full command prefix below. These are terminal commands, not SQL queries.

```powershell
cd C:\path\to\EfficientGraphBench
```

Keep original uploads in `data/Mydata/`. Prepared data defaults to `data/my_dataset/`. Results in the examples go to `experiments/my_dataset/`. Paths containing spaces must be enclosed in double quotes.

**Usual sequence:** preview → prepare → inspect → matrix → compare. Use view whenever you want a searchable data sheet. Built-in datasets do not need prepare.

In option syntax, `<value>` means replace it with your actual value; do not type the angle brackets. Options follow their command.

## General help and completion

| Syntax | Use |
|---|---|
| `.venv\Scripts\egbench --install-completion` | Install command autocompletion for the current shell. |
| `.venv\Scripts\egbench --show-completion` | Print the autocompletion script for the current shell. |
| `.venv\Scripts\egbench --help` | List all commands and general options. |
| `.venv\Scripts\egbench <command> --help` | Show all options for that command. Example: `.venv\Scripts\egbench prepare --help`. |

## hardware

Check whether PyTorch can use your GPU. Shows CUDA availability, device name, and GPU memory. Does not train anything.

### Examples

```powershell
.venv\Scripts\egbench hardware
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--help` | — | Show `hardware` help and exit. |

## datasets

List supported dataset names and identifiers. Does not download or train anything.

### Examples

```powershell
.venv\Scripts\egbench datasets
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--help` | — | Show `datasets` help and exit. |

## preview

Read an input table and show its column names and first five rows in the terminal. Use this before preparing your files.

### Examples

```powershell
.venv\Scripts\egbench preview --file data/Mydata/datatable1.csv
.venv\Scripts\egbench preview --file data/Mydata/graph.xlsx --sheet Nodes
.venv\Scripts\egbench preview --file data/Mydata/graph.pdf --page 1 --table 1
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--file <value>` | Required | Select a CSV, TSV, XLSX, or readable ruled PDF table. With view, this takes precedence over dataset selection. |
| `--sheet <value>` | Automatic / unset | Select an Excel sheet by its exact name. Required when the workbook has multiple sheets. With view, requires --file. |
| `--page <value>` | Automatic / unset | Select a PDF page, numbered from 1. With view, requires --file. |
| `--table <value>` | Automatic / unset | Select a PDF table within a page, numbered from 1. Specify page and table when extraction is ambiguous. With view, requires --file. |
| `--header-row <value>` | 1 | Select the row containing column names, numbered from 1 within the selected sheet/table. With view, requires --file. |
| `--help` | — | Show `preview` help and exit. |

## view

Create a searchable HTML data sheet containing all rows. Open the resulting HTML file in your browser; this command does not launch the browser or train a model.

### Examples

```powershell
.venv\Scripts\egbench view --file data/Mydata/datatable1.csv --output data/my-data-view.html
.venv\Scripts\egbench view --dataset custom --dataset-path data/my_dataset --output data/prepared-view.html
.venv\Scripts\egbench view --dataset Cora --output data/Cora-view.html
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--dataset <value>` | Cora | Select a built-in dataset; use custom for your own prepared dataset. |
| `--dataset-path <value>` | Automatic / unset | For custom data, select the prepared folder containing nodes.csv and edges.csv, or a supported NPZ file. Do not point to the original unprepared CSV. |
| `--file <value>` | Automatic / unset | Select a CSV, TSV, XLSX, or readable ruled PDF table. With view, this takes precedence over dataset selection. |
| `--sheet <value>` | Automatic / unset | Select an Excel sheet by its exact name. Required when the workbook has multiple sheets. With view, requires --file. |
| `--page <value>` | Automatic / unset | Select a PDF page, numbered from 1. With view, requires --file. |
| `--table <value>` | Automatic / unset | Select a PDF table within a page, numbered from 1. Specify page and table when extraction is ambiguous. With view, requires --file. |
| `--header-row <value>` | 1 | Select the row containing column names, numbered from 1 within the selected sheet/table. With view, requires --file. |
| `--data-dir <value>` | data | Choose the folder that stores downloaded built-in datasets. |
| `--columns <value>` | Automatic / unset | Choose comma-separated column names. Original tables show all columns by default; built-in/NPZ views show the first 12 features plus IDs, labels, and splits. For other features use feature_13,feature_14, etc. |
| `--output <value>` | data/dataset-view.html | Choose a new HTML filename. Existing files are not overwritten; select a different filename if it already exists. |
| `--help` | — | Show `view` help and exit. |

## prepare

Convert your node and connection tables into a validated graph dataset. Original imported tables and their column names are also preserved. If no split column exists, generate a fixed, approximately 60/20/20 training/validation/test split.

### Examples

```powershell
.venv\Scripts\egbench prepare -n data/Mydata/datatable1.csv -e data/Mydata/conn.csv
.venv\Scripts\egbench prepare -n data/Mydata/datatable1.csv -e data/Mydata/conn.csv --source from_id --target to_id
.venv\Scripts\egbench prepare --nodes data/Mydata/graph.xlsx --nodes-sheet Nodes --edges data/Mydata/graph.xlsx --edges-sheet Edges --output-dir data/excel_dataset
.venv\Scripts\egbench prepare --nodes data/Mydata/graph.pdf --nodes-page 1 --nodes-table 1 --edges data/Mydata/graph.pdf --edges-page 2 --edges-table 1 --output-dir data/pdf_dataset
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--nodes / -n <value>` | Required | Required: path to the table containing node IDs, numeric inputs, and labels. Short form: -n. |
| `--edges / -e <value>` | Required | Required: path to the connection table containing source and target IDs. Short form: -e. Both endpoints must match IDs in the node table. |
| `--output-dir <value>` | data/my_dataset | Choose a new or empty folder for prepared data; nonempty folders are protected. |
| `--node-id <value>` | Automatic / unset | Specify the exact ID column. Otherwise detect exactly one of node_id, id, person_id (case-insensitive). Multiple matches require an explicit choice. |
| `--label <value>` | Automatic / unset | Specify the correct-answer column to predict. Otherwise detect exactly one of label, category, class, target (case-insensitive). |
| `--source <value>` | Automatic / unset | Specify the source-ID column in the connection table. Otherwise detect exactly one of source, from_id, from, src (case-insensitive). |
| `--target <value>` | Automatic / unset | Specify the destination-ID column in the connection table. Otherwise detect exactly one of target, to_id, to, dst (case-insensitive). |
| `--features <value>` | Automatic / unset | Choose comma-separated numeric input columns, such as age,marks. If omitted, automatically select fully numeric columns excluding the ID, label, and split columns. Review this selection to avoid using columns that reveal the answer. |
| `--split-column <value>` | split | Name the optional split column containing train, val, test, or unused. If absent, generate splits. |
| `--split-seed <value>` | 0 | Set the seed for generated dataset splits. In prepare this controls custom splits; in run/matrix/inspect it controls Amazon/Coauthor splits. It does not regenerate an already prepared custom split. Range: 0 through 4294967295. |
| `--nodes-sheet <value>` | Automatic / unset | Select the Excel sheet name for the nodes input. The two inputs may be different sheets/tables in the same file. |
| `--edges-sheet <value>` | Automatic / unset | Select the Excel sheet name for the edges input. The two inputs may be different sheets/tables in the same file. |
| `--nodes-page <value>` | Automatic / unset | Select the PDF page number (starting at 1) for the nodes input. The two inputs may be different sheets/tables in the same file. |
| `--edges-page <value>` | Automatic / unset | Select the PDF page number (starting at 1) for the edges input. The two inputs may be different sheets/tables in the same file. |
| `--nodes-table <value>` | Automatic / unset | Select the PDF table number within the selected page (starting at 1) for the nodes input. The two inputs may be different sheets/tables in the same file. |
| `--edges-table <value>` | Automatic / unset | Select the PDF table number within the selected page (starting at 1) for the edges input. The two inputs may be different sheets/tables in the same file. |
| `--nodes-header-row <value>` | 1 | Select the column-header row number within the selected table (starting at 1) for the nodes input. The two inputs may be different sheets/tables in the same file. |
| `--edges-header-row <value>` | 1 | Select the column-header row number within the selected table (starting at 1) for the edges input. The two inputs may be different sheets/tables in the same file. |
| `--help` | — | Show `prepare` help and exit. |

Successful output contains `nodes.csv`, `edges.csv`, `original_nodes.csv`, `original_edges.csv`, and `preparation.json`. Training uses the prepared folder, not the original uploads. Replace example column names with your actual headers.

Excel formulas must first be replaced with values. PDF import requires readable ruled tables; scanned PDFs are not supported. Node labels and meaningful connections are required for testing.

## inspect

Validate the selected dataset and display its graph size, split counts, feature processing, fingerprint, and each model’s parameter count. Does not train; uncached built-in datasets may download.

### Examples

```powershell
.venv\Scripts\egbench inspect --dataset custom --dataset-path data/my_dataset
.venv\Scripts\egbench inspect --dataset Cora
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--dataset <value>` | YAML value, otherwise Cora | Select a built-in dataset; use custom for your own prepared dataset. |
| `--dataset-path <value>` | Automatic / unset | For custom data, select the prepared folder containing nodes.csv and edges.csv, or a supported NPZ file. Do not point to the original unprepared CSV. |
| `--data-dir <value>` | YAML value, otherwise data | Choose the folder that stores downloaded built-in datasets. |
| `--split-seed <value>` | YAML value, otherwise 0 | Set the seed for generated dataset splits. In prepare this controls custom splits; in run/matrix/inspect it controls Amazon/Coauthor splits. It does not regenerate an already prepared custom split. Range: 0 through 4294967295. |
| `--split-index <value>` | YAML value, otherwise 0 | Select a WikiCS official split, 0 through 19. For other datasets leave at 0. |
| `--config <value>` | Automatic / unset | Load a YAML settings file. Explicit command options override its values. See the configuration section below. |
| `--help` | — | Show `inspect` help and exit. |

A successful inspection checks data validity; it does not guarantee every model will fit in GPU memory.

## run

Train and evaluate one model with one random seed. Save metrics, settings, logs, and the best validation checkpoint.

### Examples

```powershell
.venv\Scripts\egbench run --dataset custom --dataset-path data/my_dataset --model GraphSAGE --device cuda --output-dir experiments/my_dataset
.venv\Scripts\egbench run --dataset Cora --model Graphormer --device cuda --output-dir experiments/cora
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--config <value>` | Automatic / unset | Load a YAML settings file. Explicit command options override its values. See the configuration section below. |
| `--dataset <value>` | YAML value, otherwise Cora | Select a built-in dataset; use custom for your own prepared dataset. |
| `--model <value>` | YAML value, otherwise GCN | Select one of the 10 model names listed below. Names are case-insensitive. |
| `--seed <value>` | YAML value, otherwise 42 | Set the random seed for model initialization and training; does not change the fixed custom train/test split. Range: 0 through 4294967295. |
| `--epochs <value>` | YAML value, otherwise 200 | Number of training epochs, at least 1. Each epoch processes the full graph. Short runs check execution but do not establish final model performance. |
| `--device <value>` | YAML value, otherwise auto | Use cuda for GPU, cpu for CPU, or auto to select GPU when available. cuda fails when CUDA is unavailable. |
| `--output-dir <value>` | YAML value, otherwise experiments | Choose the experiment folder. For compare, use the same folder as training. |
| `--dataset-path <value>` | Automatic / unset | For custom data, select the prepared folder containing nodes.csv and edges.csv, or a supported NPZ file. Do not point to the original unprepared CSV. |
| `--split-seed <value>` | YAML value, otherwise 0 | Set the seed for generated dataset splits. In prepare this controls custom splits; in run/matrix/inspect it controls Amazon/Coauthor splits. It does not regenerate an already prepared custom split. Range: 0 through 4294967295. |
| `--split-index <value>` | YAML value, otherwise 0 | Select a WikiCS official split, 0 through 19. For other datasets leave at 0. |
| `--help` | — | Show `run` help and exit. |

## matrix

Train and evaluate all 10 models for each requested seed. The default three seeds mean 30 runs. Print the latest comparison and save its Markdown report.

### Examples

```powershell
.venv\Scripts\egbench matrix --dataset custom --dataset-path data/my_dataset --device cuda --output-dir experiments/my_dataset
.venv\Scripts\egbench matrix --dataset Cora --seeds 42 --epochs 2 --device cuda --output-dir experiments/smoke
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--config <value>` | Automatic / unset | Load a YAML settings file. Explicit command options override its values. See the configuration section below. |
| `--dataset <value>` | YAML value, otherwise Cora | Select a built-in dataset; use custom for your own prepared dataset. |
| `--seeds <value>` | 42,43,44 | Comma-separated training seeds, for example 42,43,44. Duplicate seeds are removed. Overrides the single seed in YAML. |
| `--epochs <value>` | YAML value, otherwise 200 | Number of training epochs, at least 1. Each epoch processes the full graph. Short runs check execution but do not establish final model performance. |
| `--device <value>` | YAML value, otherwise auto | Use cuda for GPU, cpu for CPU, or auto to select GPU when available. cuda fails when CUDA is unavailable. |
| `--output-dir <value>` | YAML value, otherwise experiments | Choose the experiment folder. For compare, use the same folder as training. |
| `--dataset-path <value>` | Automatic / unset | For custom data, select the prepared folder containing nodes.csv and edges.csv, or a supported NPZ file. Do not point to the original unprepared CSV. |
| `--split-seed <value>` | YAML value, otherwise 0 | Set the seed for generated dataset splits. In prepare this controls custom splits; in run/matrix/inspect it controls Amazon/Coauthor splits. It does not regenerate an already prepared custom split. Range: 0 through 4294967295. |
| `--split-index <value>` | YAML value, otherwise 0 | Select a WikiCS official split, 0 through 19. For other datasets leave at 0. |
| `--help` | — | Show `matrix` help and exit. |

A new matrix replaces active results for the same dataset in that output folder as its results are written. Old logs and checkpoints remain. Use a separate output folder to retain a separate comparison. A single run replaces its previous dataset/model/seed result.

## compare

Read saved results and display the latest comparison without training. GNNs and MLP appear first, then a horizontal divider, then the graph Transformers.

### Examples

```powershell
.venv\Scripts\egbench compare --dataset custom --output-dir experiments/my_dataset
.venv\Scripts\egbench compare --dataset custom --output-dir experiments/my_dataset --report
```

### Every option

| Option syntax | Default | Use |
|---|---|---|
| `--dataset <value>` | cora | Select a built-in dataset; use custom for your own prepared dataset. |
| `--output-dir <value>` | experiments | Choose the experiment folder. For compare, use the same folder as training. |
| `--report / --no-report` | False | With --report, write or replace reports/<dataset-id>.md inside the output folder. --no-report only displays the table. |
| `--help` | — | Show `compare` help and exit. |

Accuracy is a fraction (`0.95` means 95%); SD measures variation across runs. Train s is training time in seconds; Latency ms is full-graph inference time in milliseconds; GPU MiB is measured peak PyTorch GPU allocation. Failed/OOM runs are excluded from averages.

## Models

Graphormer, GraphGPS, GCN, GraphSAGE, GAT, SGFormer, MLP, SGC, APPNP, GATv2.

Graphormer is a node-classification adaptation with a custom EfficientGraphBench 4,096-node safeguard (not an architectural Graphormer limit). GraphGPS is a simplified baseline. MLP uses features only, without connections.

## Built-in dataset names

- Cora
- CiteSeer
- PubMed
- Amazon Computers
- Amazon Photo
- Coauthor CS
- Coauthor Physics
- WikiCS
- Flickr
- ogbn-arxiv

## YAML configuration: settings without dedicated command options

Save this as `configs/my_experiment.yaml`. Paths inside it are relative to the project folder when commands run there.

```yaml
dataset: custom
dataset_path: data/my_dataset
data_dir: data
output_dir: experiments/my_dataset
device: auto
seed: 42
threads: 4
split_seed: 0
split_index: 0
model:
  name: gcn
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5
  heads: 4
  propagation_steps: 10
  alpha: 0.1
training:
  epochs: 200
  lr: 0.01
  weight_decay: 0.0005
  latency_warmup: 10
  latency_repeats: 50
```

```powershell
.venv\Scripts\egbench run --config configs/my_experiment.yaml
.venv\Scripts\egbench matrix --config configs/my_experiment.yaml
```

Top-level keys have the meanings of the same command options above. `threads` sets the positive number of CPU threads. Matrix ignores `model.name` and runs every registered model; its `--seeds` controls training seeds.

| YAML setting | Use |
|---|---|
| `model.hidden_dim` | Hidden representation width. Positive integer; must be divisible by heads for GAT, GATv2, Graphormer, and GraphGPS. SGC does not use it. |
| `model.num_layers` | Model depth, at least 2. For SGC, sets graph-propagation steps. SGFormer uses it for its GNN component. |
| `model.dropout` | Training dropout probability, 0 to less than 1. SGC does not use it. |
| `model.heads` | Positive attention-head count for GAT, GATv2, Graphormer, and GraphGPS. SGFormer currently uses one Transformer head. |
| `model.propagation_steps` | APPNP propagation iterations; positive integer. |
| `model.alpha` | APPNP teleport probability; greater than 0 and at most 1. |
| `training.epochs` | Positive training epoch count. Same as --epochs. |
| `training.lr` | Positive Adam learning rate. |
| `training.weight_decay` | Nonnegative Adam weight decay. |
| `training.latency_warmup` | Number of untimed inference warm-up passes, at least 0. |
| `training.latency_repeats` | Number of timed inference passes, at least 1. |

These YAML keys are not standalone CLI flags: use `--config` to set them.

Failure statuses: SUCCESS, OOM, UNSUPPORTED_GRAPH_SIZE, CONFIG_ERROR, RUNTIME_ERROR. See GRAPH_SIZE_SCALABILITY_REVIEW.md for measured evidence and implementation limits.

## scale --step

Syntax: `egbench scale --dataset PubMed --step 2000 --seeds 42,43,44 --epochs 200 --device cuda --output-dir experiments/pubmed-scaling`

`--step INTEGER`: positive increment in graph nodes; overrides `--sizes`. Tests increasing sizes and includes the full graph. `--models` selects comma-separated models; the default is all primary models. Each run starts fresh on an induced subgraph. After OOM, skips that model's remaining seeds and larger sizes. Displays the last successful size/result and first observed OOM size. Full results: `scaling-results.json`; last successes, OOMs and skipped jobs: `scaling-summary.json`.

`matrix`: after a model reports OOM, its remaining seeds are skipped, recorded in `matrix-skipped.json`. Skipped runs are not reported as measured OOMs.
