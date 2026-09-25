# EfficientGraphBench 0.1.0

EfficientGraphBench measures graph-learning models on your hardware: predictive accuracy, preprocessing, training, inference latency and memory. It provides a Python library and the `egbench` terminal command.

It exists to help researchers choose practical implementations before committing to their software and compute requirements. Results compare usable implementations and their recorded recipes; they are not automatically architecture-only comparisons.

## Install

From this checkout (Python 3.11+):

```bash
python -m pip install -e .
```

From the built wheel:

```bash
python -m pip install dist/efficientgraphbench-0.1.0-py3-none-any.whl
```

The package has not been published to PyPI. `pip install efficientgraphbench` is the intended future installation, not a verified public release. A project license must be selected before public release; third-party APPNP source retains the included PyG MIT license.

Core dependencies include PyTorch and PyTorch Geometric. Install the PyTorch build appropriate for your CPU/CUDA system. Extras: `[tables]` for Excel/PDF tables, `[ogb]` for OGB datasets, `[setup]` for uv, `[dev]` for tests/build tools. Research model stacks are installed separately.

New computer or notebook? Follow [YOUR_NOTEBOOK_SETUP.md](YOUR_NOTEBOOK_SETUP.md).

## Quick start

```bash
egbench models
egbench run --dataset cora --model gcn
egbench matrix --dataset cora --seeds 1,2,3
```

`--seeds` always lists literal seed IDs: `--seeds 3` runs seed 3 once. Defaults are 42,43,44. Existing `scale`, CSV preparation, viewing, inspection, comparison, recommendation and provenance commands remain available. Use `egbench --help` or `egbench COMMAND --help`.

## Python

```python
from efficientgraphbench import Benchmark

results = Benchmark(
    dataset="cora",
    models=["gcn", "graphsage"],
    seeds=[1, 2, 3],
    device="auto",
    epochs=200,
).run()
print(results.summary())
results.save_csv("results.csv")
results.save_json("results.json")
results.save_markdown("report.md")
```

Single model:

```python
result = Benchmark.run_model(model="gcn", dataset="cora")
print(result.status, result.accuracy, result.training_time)
```

Notebook recipes: [PYTHON_COMMANDS.md](PYTHON_COMMANDS.md). Full Python and CLI reference: [PYTHON_API_REFERENCE.md](PYTHON_API_REFERENCE.md).

## Reference environments

```bash
python -m pip install uv
egbench setup graphgps
egbench setup sgformer
egbench environments
egbench run --dataset cora --model graphgps --device cuda
```

Git is also required. Setup checks pinned commits and verifies imports. The library launches the correct isolated interpreter automatically. Installation readiness is not a guarantee that a dataset fits memory. Current reproducible research installation locks support Windows x86-64 with Python 3.10 and PyTorch 1.13.1/CUDA 11.7. Other platforms can use the core models; reference setup on other platforms remains unsupported pending validation.

Downloaded environments use the platform user cache, overridable by `EGBENCH_HOME`. Existing development installations in this checkout remain usable. Installed package files are never used as a writable cache.

## Models and tasks

Current task: homogeneous, transductive node classification on a single graph.

- PyG: GCN, GraphSAGE, GAT, GATv2, SGC; optional SGFormer (PyG).
- Reference implementations: APPNP (bundled unchanged PyG citation class), GraphGPS and SGFormer (isolated official repositories).
- Feature baseline: MLP.
- Experimental: APPNP (Adapted), GraphGPS (Adapted), Graphormer (Adapted).
- Official Graphormer remains visible with `UNSUPPORTED_TASK`; no validated faithful citation-node-classification adapter is claimed. `egbench models graphormer` explains this.

`models="all"` selects the primary set including MLP and the Graphormer task gate. Experimental variants require explicit selection. `sgformer_reference` is an alias for the same reference implementation as `sgformer`.

## Datasets and custom input

Use `egbench datasets` for the built-in list. The dataset interface also accepts user graphs; Cora and PubMed are examples, not hardcoded architecture requirements.

```python
from efficientgraphbench.datasets import load_dataset

data = load_dataset("custom", nodes="nodes.csv", edges="edges.csv")
results = Benchmark(dataset=data, models=["gcn"], seeds=[42]).run()
```

`nodes.csv`: unique `node_id`, numeric `feature_*` columns, `label`, and `split` (`train`, `val`, `test`, or `unused`). `edges.csv`: `source`, `target`, optionally finite numeric edge-attribute columns. Features and endpoints are validated. Keeping edge attributes does not mean every model consumes them. NPZ archives and existing `prepare` workflows remain supported.

## Metrics and failure handling

Reports include accuracy, validation accuracy, preprocessing seconds, training seconds, warmed synchronized latency mean/median/standard deviation in milliseconds, peak GPU allocated/reserved MiB, sampled process RAM MiB, parameter counts and actual epochs. RAM means process RSS, not machine-wide usage.

Failures remain results: OOM, RESOURCE_LIMIT, UNSUPPORTED_TASK, UNSUPPORTED_GRAPH_SIZE, DEPENDENCY_ERROR, ENVIRONMENT_NOT_INSTALLED, CONFIG_ERROR, PREPROCESSING_ERROR, RUNTIME_ERROR. Skipped seeds are distinguished from measured failures. Pretraining deterministic failures skip remaining seeds; the configured OOM policy also skips remaining seeds after training OOM.

## Scaling

```bash
egbench scaling --dataset pubmed --model gcn --step 2000 --epochs 200 --device cuda
```

Fresh models train on successively larger induced subgraphs with original split membership. Stop each model after OOM, retain the last successful result, and continue other models. Display only the final result per model. Detailed runs remain in files. This is not mini-batch training; accuracy at different sizes concerns different prediction problems.

## Reproducibility and limits

Records include version, Git commit when available, source hashes, reference commit, software, environment specifications, dataset/split fingerprints, seeds, model configuration, hardware and profiler settings. A wheel installed without Git correctly records no project commit. GPU capacity is measured, never hardcoded.

Reference recipes and dependencies differ. Dense preprocessing can time out or exhaust RAM before GPU training. Graph-level prediction, heterogeneous graphs and link prediction are future extensions. No web dashboard or Docker deployment is included.

## Development

```bash
python -m pip install -e ".[dev,tables]"
python -m pytest -q
python -m build
```

See [docs/getting_started.md](docs/getting_started.md), [docs/architecture.md](docs/architecture.md), and [CHANGELOG.md](CHANGELOG.md).

Generated results, personal notebooks, datasets and development notes are excluded from this source repository. Run the documented commands to generate results on your own machine.

Linux update: GraphGPS now has a Linux x86-64 installation path; see [GraphGPS on Linux](docs/graphgps_linux.md). It requires verification on the target server. SGFormer automatic setup remains Windows-only.
