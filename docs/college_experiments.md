# College GPU experiments

Run these commands **inside your SSH session** on the college Ubuntu computer, from your existing EfficientGraphBench folder. Python then uses the college GPU, even though you type on your laptop.

```bash
git pull
source .venv/bin/activate
python -m pip install -e '.[ogb]'
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run from Python

```bash
python examples/college_benchmark.py
```

The script runs OGB-Arxiv on seven core models with **10 seeds, 200 epochs each**, and saves `summary.csv`, `runs.csv`, and `runs.json` under `experiments/college-ogbn-arxiv-fixed-10/`. Downloads happen on first use. Edit the four settings at the top of the script to change datasets, models, run count, or splits. Set `RUNS = 20` for 20 runs. No large benchmarks were executed as part of adding this feature.

Equivalent Python code (also works in a notebook whose kernel runs on the server):

```python
from efficientgraphbench import Benchmark

results = Benchmark(
    dataset="ogbn-arxiv",
    models=["gcn", "graphsage"],
    runs=10,
    device="cuda",
    epochs=200,
    output_dir="experiments/college-arxiv",
).run()
display(results.summary())  # In a notebook; use print(...) in a script.
results.save_json("experiments/college-arxiv/runs.json")
```

## Choose the experiment

| Setting | What it runs |
| --- | --- |
| `runs=10` / `runs=20` | Training seeds 1–10 / 1–20 |
| `splits="fixed"` (default) | Same dataset split for every seed; preserves official OGB splits |
| `splits="official"` | Cycles available official folds; 10 heterophilous folds, or 20 WikiCS folds; datasets with one split retain it |
| `splits="random"` | New stratified 60/20/20 split per seed; identical splits across models; **not an official OGB evaluation** |
| `seeds=[...]` | Explicit seeds instead of `runs`; cannot combine both |

For heterophilous datasets, use `splits="official", runs=10`; for 20 runs each of the ten folds is used twice with different training seeds. Summaries show successful run count, attempted count, distinct successful splits, accuracy mean and sample standard deviation. Failed/skipped runs are excluded from averages and remain in the detailed records. Split hashes and configurations are saved. `results.summary(across_splits=False)` shows separate fold groups.

Terminal shortcuts:

```bash
egbench repeat --dataset ogbn-arxiv --runs 10 --device cuda --output-dir experiments/arxiv-10
egbench repeat --dataset roman-empire --runs 20 --splits official --device cuda --output-dir experiments/roman-20
egbench repeat --help
```

## Datasets

OGB: **ogbn-arxiv**, **ogbn-products**.

Added heterophilous datasets: **roman-empire**, **amazon-ratings**, **actor**, **texas**, **wisconsin**.

These are homogeneous, single-label node classification datasets evaluated with accuracy. OGB-Arxiv retains its time split; OGB-Products retains its sales-rank split. Products' already-undirected edges are retained without another full coalescing copy. Other OGB tasks (graph prediction, link prediction, heterogeneous MAG, multilabel Proteins) are not supported by this node classification harness. Minesweeper, Tolokers and Questions are not added here because their standard evaluation requires ROC-AUC support.

Sources: [OGB node datasets](https://ogb.stanford.edu/docs/nodeprop/), [PyG heterophilous datasets](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.HeterophilousGraphDataset.html).

## Detailed measurements

`runs.json` contains per-run measurements; `runs.csv` stores nested fields as JSON text.

| Field | Meaning |
| --- | --- |
| `phase_times_sec` | Sequential preprocessing, initialization/transfer, training, checkpoint/test, inference phases; worker phase names differ where operations are combined |
| `optimization_time_sec`, `epoch_times_sec` | Forward/loss/backward/optimizer together, total and per epoch |
| `validation_time_sec` | Validation forward and accuracy across epochs |
| `checkpoint_write_time_sec` | Writing the selected checkpoint |
| `checkpoint_restore_and_test_time_sec` | Restoring the checkpoint and evaluating the test split |
| `inference_latency_*_ms` | Warmed forward latency statistics |
| `memory_snapshots` | Process RSS, allocated/reserved GPU memory at phase boundaries; GPU peak in each snapshot is **cumulative**, not a phase-local peak |
| `parameter_memory_mib`, `gradient_memory_mib` | Parameter and surviving gradient tensor bytes |
| `optimizer_tensor_memory_mib`, `graph_tensor_memory_mib` | Optimizer-state tensor bytes and graph tensor bytes; logical tensor sizes, not total allocator usage |
| `peak_gpu_allocated_mib`, `peak_gpu_reserved_mib`, `peak_cpu_memory_mb` | Overall allocator peaks and sampled process RSS peak |

Memory values are MiB (including the legacy `peak_cpu_memory_mb` field). Tensor categories do not include temporary activations, allocator overhead or all Python objects, and are not an additive breakdown of peak memory. GPU operations are synchronized at timing boundaries. CPU RSS sampling every 10 ms can miss brief peaks. External models report worker-process memory; controller memory is not included in that peak. Download/cache state affects preprocessing time. Training time includes validation and best-state copies but excludes final checkpoint writing and test evaluation.

Fixed-split repetitions share a checksummed exported graph in `graph-cache/` under the output directory. Different splits need separate exports. Checkpoints still consume disk per successful run; allow space for the downloaded dataset, graph exports, and model checkpoints. Experiments run sequentially; use separate output directories for concurrent jobs.

## Large graph limits

Loading support does **not** imply a graph will fit every model. Current training is full-batch, including OGB-Products (millions of nodes). GraphGPS uses dense Laplacian preprocessing and dense attention; even a 4090 may fail on large graphs. This update does not add neighbor sampling or change model architectures. Run Arxiv first; use the existing scaling benchmark for capacity experiments. OOM stops remaining seeds for that model; such experiments are incomplete and must not be reported as ten successful runs. Official Graphormer remains unsupported for this node classification task. Reference SGFormer/GraphGPS need their separate environments; the script defaults to core models available in the main environment.

## Use a notebook on the college GPU

In the **college SSH terminal**:

```bash
source .venv/bin/activate
python -m pip install jupyterlab ipykernel ipywidgets
python -m ipykernel install --user --name egbench-college --display-name "EfficientGraphBench College GPU"
python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8888
```

In a **second laptop terminal**, replace `COLLEGE_HOST` with the same host/IP you use for SSH:

```bash
ssh -N -L 8888:127.0.0.1:8888 ise@COLLEGE_HOST
```

Open the `http://127.0.0.1:8888/lab?token=...` URL printed by Jupyter in your laptop browser. Select **EfficientGraphBench College GPU** as the notebook kernel. Use the Python example above. Keep the SSH tunnel open. A normal local notebook kernel still uses your laptop hardware.
