# Reference benchmarks

Run commands from the EfficientGraphBench folder with its usual environment active.
You do not activate the model-specific environments yourself.

```powershell
egbench models
egbench run --dataset cora --model graphgps --device cuda
egbench run --dataset cora --model sgformer --device cuda
egbench matrix --dataset cora --device cuda
egbench matrix --dataset pubmed --device cuda
egbench matrix --dataset custom --dataset-path data/my_dataset --device cuda
```

`matrix` defaults to seeds 42,43,44 and 200 maximum epochs. Select models with
`--models gcn,graphgps,sgformer`. Use `--epochs` to change the explicit budget.
`--output-dir experiments/my-comparison` keeps a separate comparison.
All models execute sequentially on the same machine. The actual GPU here reports
6141 MiB, not 8 GB. Model failures remain in the report; a matrix with failures
returns exit code 1 after attempting every model.

## Implementations

| Model ID | Implementation |
|---|---|
| gcn, graphsage, gat, gatv2 | Maintained PyG model classes |
| sgc | PyG SGConv |
| appnp | Unchanged Net class from PyG's pinned citation benchmark, using PyG APPNP |
| appnp_adapted | Previous local feature MLP plus APPNP propagation, experimental |
| mlp | Existing feature baseline, excluded from primary model recommendations |
| graphgps | Official GraphGPS GPSModel in its isolated environment |
| sgformer, sgformer_reference | Authors' medium/ours.py SGFormer plus models.GCN |
| sgformer_pyg | Previous maintained PyG SGFormer, explicitly named |
| graphgps_adapted | Previous local SAGE/Performer implementation, experimental |
| graphormer_adapted | Previous local Graphormer-inspired model, experimental |
| graphormer | UNSUPPORTED_TASK for the current continuous-feature node benchmark |

GraphGPS uses the repository's Actor node-classification recipe: GCN+Transformer,
two layers, width64, four heads, LapPE, GELU, AdamW, cosine warmup and gradient clipping.
These are source-supported configuration choices; this is not a tuned Cora recipe
or a claim to reproduce paper accuracy. Original GPSModel, GPSLayer, LapPE encoder,
node head and positional preprocessing are imported unchanged.
Its Planetoid input uses raw features, as in the upstream loader.

SGFormer uses the authors' Cora/PubMed optimizer groups, depth4 graph branch,
one transformer layer, graph weight0.8, and dataset-specific learning rate/dropout.
Raw features match the published --no_feat_norm setting; edges are symmetrized as
in their main.py. The actual parser distinguishes --use_residual from
--ours_use_residual: the provided commands leave the latter false, which is
preserved and recorded. Patience200 is retained. The common 200 maximum-epoch
default is explicit; the authors' parser defaults to500. Canonical public masks
replace random splits, and test accuracy is evaluated at the best validation
checkpoint. No attention or branch code is rewritten.

## Model-specific settings

Reference workers use their source recipe rather than `model.hidden_dim` or
`training.lr` from the old common-budget configuration. Override supported settings
explicitly using `model_options`; the actual resolved recipe is saved in each result.

```yaml
dataset: cora
model:
  name: graphgps
device: cuda
training:
  epochs: 200
model_options:
  lr: 0.0005
  layers: 2
  hidden_dim: 64
worker_timeout_sec: 7200
```

GraphGPS options: lr, weight_decay, layers, hidden_dim, heads, dropout.
SGFormer options: hidden_dim, layers, heads, dropout, trans_layers, trans_dropout,
graph_weight, alpha, lr, weight_decay, trans_weight_decay, use_bn, use_residual,
use_weight, use_act, patience. Unknown options must not be silently ignored.

## Results and reproducibility

The output directory contains raw JSON/CSV, configs, logs, checkpoints, reports,
and `exchange/<run-id>/`. Each external exchange contains request.json, graph.npz,
split.json, worker.log, result.json and checkpoint.pt on success. NPZ uses numeric
arrays without pickle. JSON splits contain canonical node indices and a SHA256.
Raw and normalized features are transported separately when available; only the
selected model input is moved to the GPU.

Comparison groups require the same physical hardware, canonical graph/split,
task, metric and profiler settings. They intentionally permit different framework
versions and architecture-specific recipes. Experimental results have a separate
comparison category. Historical schema3 results are not primary reference evidence.

Latency uses eval/inference mode, warmup, repeated synchronized full-graph forwards,
with mean, median and standard deviation saved. GPU allocated and reserved peaks
are separate allocator metrics; neither is NVML total process VRAM. CPU RSS is
sampled every10ms; in-process results include the controller's resident memory,
whereas reference results measure a fresh worker. GPU peaks include preprocessing,
initialization, training and inference. Model preprocessing cost is visible.
Training includes validation and best-checkpoint CPU copies. These measurements
describe real implementation costs, not pure architectural complexity.

There is no Git repository in this workspace, so the project Git commit is null.
Source snapshot hashes, exact reference commits, complete installed package locks,
resolved recipes, hardware identity, dataset/split fingerprints and profiler settings
are stored instead. Do not interpret null as a known source commit.

```powershell
egbench compare --dataset cora --output-dir experiments/my-comparison --report
egbench provenance
egbench recommend --dataset cora --output-dir experiments/my-comparison --minimum-accuracy 0.8 --maximum-latency-ms 10 --maximum-gpu-mib 6141
egbench scale --dataset cora --sizes 500,1000,2708 --models gcn,sgformer,graphgps --device cuda
```

Recommendations use complete measured primary successes, mean accuracy/median
latency across seeds, and maximum reserved GPU MiB across seeds. They do not infer
memory feasibility on another GPU. Scaling uses deterministic induced subgraphs
and original split membership; reduced graphs are different prediction problems.
Default scaling budget is20 epochs, seeds42,43,44; `scaling-results.json` records
all failures. This is a measured sweep, not a claimed architectural maximum.

## Installing on a new Windows machine

Install the controller normally, then install Git and uv. Run:

```powershell
python scripts/setup_reference.py graphgps
python scripts/setup_reference.py sgformer
python scripts/setup_appnp.py
```

The setup script creates separate Python3.10 environments from package locks and
pinned clean repositories. It does not downgrade the controller environment.
The Windows launcher paths are recorded in `environments/*/metadata.json`.
Graphormer's environment is intentionally not installed while its task gate is closed.

## Graphormer task gate

For these continuous-feature citation graphs the classification is **C: additional
input/architecture adaptation required**, not merely a graph-token readout change.
The official input embeds integer atom categories, embeds edge/path categories,
adds degree encodings and a graph token; the task pipeline targets graph prediction.
Changing the continuous feature encoder as well as the task head has not been
validated as an unchanged reference architecture here. Therefore the primary ID
returns UNSUPPORTED_TASK; this does not claim node prediction is theoretically
impossible with Graphormer. The existing4096 guard belongs only to graphormer_adapted.

Sources:
- https://github.com/rampasek/GraphGPS/tree/28015707cbab7f8ad72bed0ee872d068ea59c94b
- https://github.com/qitianwu/SGFormer/tree/3578e101c701491ce068bf26a9b029d2134903be
- https://github.com/microsoft/Graphormer/blob/main/graphormer/modules/graphormer_layers.py
- https://github.com/microsoft/Graphormer/blob/main/graphormer/models/graphormer.py
- https://github.com/pyg-team/pytorch_geometric/blob/79d33965a40b7fa83616a9f598a0f8619f25d939/benchmark/citation/appnp.py

APPNP loads only the original class definition to avoid the reference script's
top-level argument parsing, dataset download and 100-run training loop. The class
body is unchanged and verified against a pinned normalized-text SHA256. It retains
input dropout, Linear/ReLU, hidden dropout, Linear, PyG APPNP propagation, and
log_softmax. The benchmark harness supplies canonical masks and explicit budgets;
the source harness's loss-window early stopping is replaced by the configured
validation-accuracy patience (disabled by default). This changes the experiment
protocol, not the architecture; both are recorded.
