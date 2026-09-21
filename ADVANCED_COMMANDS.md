# EfficientGraphBench command reference

Run these commands in PowerShell from the project directory. Each benchmark command
starts real training; run the examples you need, not the whole file at once.

```powershell
cd C:\path\to\EfficientGraphBench
```

The existing `.venv` is ready. Activation is unnecessary with the commands below.
Model names are **GCN**, **GraphSAGE**, **GAT**, and **SGFormer**. Dataset names are
listed by `egbench datasets`. CLI/YAML names accept either public capitalization or lowercase.
Machine identifiers in saved configurations, filenames, and the raw `model`/`dataset`
columns remain lowercase; new records also include correctly capitalized display names.

## Help and available options

```powershell
.venv\Scripts\egbench --help
.venv\Scripts\egbench run --help
.venv\Scripts\egbench matrix --help
.venv\Scripts\egbench compare --help
```

| Command | Options |
|---|---|
| `run` | `--config`, `--dataset`, `--dataset-path`, `--model`, `--seed`, `--epochs`, `--device`, `--output-dir`, `--split-seed`, `--split-index` |
| `matrix` | `--config`, `--dataset`, `--dataset-path`, `--seeds`, `--epochs`, `--device`, `--output-dir`, `--split-seed`, `--split-index` |
| `compare` | `--dataset`, `--output-dir`, `--report` / `--no-report` |
| `datasets` | Lists available datasets; no downloads |
| `inspect` | `--config`, `--dataset`, `--dataset-path`, `--data-dir`, `--split-seed`, `--split-index`; checks data and model sizes without training |
| `prepare` | `--nodes`, `--edges`, `--node-id`, `--label`, `--source`, `--target`, `--features`, `--split-column`, `--split-seed`, `--output-dir`; converts CSVs |
| `hardware` | Shows PyTorch/CUDA availability, GPU name, and memory |

Defaults: Cora, GCN for a single run, 200 epochs, seed 42, automatic device selection,
and `experiments` as the output directory. A matrix runs all ten models with seeds
42, 43, and 44. Explicit CLI options override YAML; unspecified options use the YAML
or package defaults. There are no `recommend` or dashboard commands yet.

## Run each model on Cora

```powershell
.venv\Scripts\egbench run --dataset Cora --model GCN
.venv\Scripts\egbench run --dataset Cora --model GraphSAGE
.venv\Scripts\egbench run --dataset Cora --model GAT
.venv\Scripts\egbench run --dataset Cora --model SGFormer
```

Choose a seed, epoch budget, device, and separate results directory:

```powershell
.venv\Scripts\egbench run --dataset Cora --model GCN --seed 123 --epochs 100 --device cpu --output-dir experiments/cora-seed123
```

Quick installation check (two epochs; not an accuracy benchmark):

```powershell
.venv\Scripts\egbench run --dataset Cora --model GCN --epochs 2 --device cpu --output-dir experiments/smoke
```

## Run all ten models across seeds

```powershell
.venv\Scripts\egbench matrix --dataset Cora --seeds 42,43,44 --epochs 200 --device cpu --output-dir experiments/cora-cpu
```

One seed or a quick four-model check:

```powershell
.venv\Scripts\egbench matrix --dataset Cora --seeds 42 --epochs 200 --output-dir experiments/cora-one-seed
.venv\Scripts\egbench matrix --dataset Cora --seeds 42 --epochs 2 --device cpu --output-dir experiments/smoke-matrix
```

## Use configuration files

```powershell
.venv\Scripts\egbench run --config configs/experiments/cora.yaml --model SGFormer
.venv\Scripts\egbench matrix --config configs/experiments/cora.yaml --seeds 42,43,44
.venv\Scripts\egbench run --config configs/experiments/pubmed.yaml --model GraphSAGE
.venv\Scripts\egbench matrix --config configs/experiments/pubmed.yaml --seeds 42,43,44
```

Edit YAML for settings without CLI flags: `model.hidden_dim`, `model.num_layers`,
`model.dropout`, `model.heads` (GAT), `training.lr`, `training.weight_decay`,
`training.latency_warmup`, `training.latency_repeats`, `threads`, and `data_dir`.
GAT's hidden dimension must be divisible by its head count. SGFormer uses a fixed
single global attention layer/head and the configured number of local layers.

## Run another supported dataset: PubMed

PubMed is already wired into the loader. It downloads on first use and is cached
under `data/pubmed`. Its loader has not yet been validated with a real benchmark4

on this machine; start with a short run:

```powershell
.venv\Scripts\egbench run --dataset PubMed --model GCN --epochs 2 --device cpu --output-dir experiments/pubmed-smoke
```

Then run an individual model or the full comparison:

```powershell
.venv\Scripts\egbench run --dataset PubMed --model GraphSAGE --epochs 200 --output-dir experiments/pubmed
.venv\Scripts\egbench run --dataset PubMed --model GAT --epochs 200 --output-dir experiments/pubmed
.venv\Scripts\egbench run --dataset PubMed --model SGFormer --epochs 200 --output-dir experiments/pubmed
.venv\Scripts\egbench matrix --dataset PubMed --seeds 42,43,44 --epochs 200 --output-dir experiments/pubmed
.venv\Scripts\egbench compare --dataset PubMed --output-dir experiments/pubmed --report
```

Use the individual runs OR the matrix to avoid duplicating trials in one directory.
Dataset downloads require internet access. PubMed is larger than Cora and full-graph
training can require more time and memory.

## Compare saved results and export reports

View the original completed Cora benchmark:

```powershell
.venv\Scripts\egbench compare --dataset Cora
```

View another result directory and generate/update its Markdown report:

```powershell
.venv\Scripts\egbench compare --dataset Cora --output-dir experiments/cora-cpu --report
.venv\Scripts\egbench compare --dataset PubMed --output-dir experiments/pubmed --report
```

`compare` reads existing results without training. `--report` overwrites the generated
report for that dataset; keep personal notes in a separate file. `matrix` exports a
report automatically. Different hardware/settings form separate comparison groups.
Repeated runs append to the history; use a new output directory for independent studies.

| Output under your chosen directory | Contents |
|---|---|
| `raw/results.csv` | Spreadsheet-friendly metrics |
| `raw/results.jsonl` | Complete canonical run history |
| `raw/<run_id>.json` | Individual run record |
| `configs/<run_id>.yaml` | Effective configuration |
| `checkpoints/<run_id>.pt` | Best validation checkpoint |
| `logs/<run_id>.log` | Training progress/errors |
| `reports/cora.md` or `reports/pubmed.md` | Generated comparison report |

## CPU and CUDA

```powershell
.venv\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.venv\Scripts\egbench run --dataset Cora --model GCN --device cpu
.venv\Scripts\egbench run --dataset Cora --model GCN --device auto
```

This environment now has CUDA-enabled PyTorch 2.14.0+cu130 for the RTX 4050. Use:

```powershell
.venv\Scripts\egbench run --dataset Cora --model GCN --device cuda --output-dir experiments/cora-cuda
.venv\Scripts\egbench matrix --dataset Cora --device cuda --output-dir experiments/cora-cuda-matrix
```

Explicit `cuda` fails when unavailable. `auto` falls back to CPU. CPU runs have no
GPU memory measurement. Run benchmark processes sequentially for comparable results.

## What about CiteSeer, ogbn-arxiv, or my own dataset?

There are now 10 built-in dataset adapters plus custom CSV/NPZ import. See
[DATASETS.md](DATASETS.md) for the complete list, formats, split protocols, and examples.

To add a further built-in dataset as a developer:

1. Add a loader in `src/efficientgraphbench/datasets/registry.py` (or a dedicated module).
2. Return a `DatasetBundle` containing a PyG `Data` object and the class count.
   The graph must contain floating-point `x` shaped `[num_nodes, num_features]`,
   integer `edge_index` shaped `[2, num_edges]`, `torch.long` labels `y` shaped
   `[num_nodes]` with contiguous class IDs from zero, and disjoint, nonempty boolean
   `train_mask`, `val_mask`, and `test_mask` shaped `[num_nodes]`.
3. Register it in `src/efficientgraphbench/datasets/catalog.py`. Names and configuration
   validation use that registry automatically. Add a YAML experiment configuration.
4. Preserve official splits where provided; document feature preprocessing and
   whether edges are directed or converted to undirected. Set an accurate split
   label and update comparison protocol metadata/report text for the new loader.
5. Test loading, split integrity, label shape/range, and all model paths before
   running a multi-seed benchmark.

CiteSeer and ogbn-arxiv now have adapters. The OGB adapter uses official split indices
and flattens labels to `[N]`. CSV imports map node IDs and string labels automatically.
Larger graphs may require sampling or memory changes.
The current trainer supports single-label node classification;
link prediction, graph classification, and regression need trainer/metric changes.

## Developer checks

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff check src tests
.venv\Scripts\ruff format --check src tests
```
