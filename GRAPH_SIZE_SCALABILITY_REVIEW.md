# Graph-size and scalability audit

**Historical pre-migration audit.** Here, GraphGPS means the implementation now named
`graphgps_adapted`, Graphormer means `graphormer_adapted`, and SGFormer means
`sgformer_pyg`. Current reference implementations and validation outcomes are documented
in [REFERENCE_MIGRATION_STATUS.md](REFERENCE_MIGRATION_STATUS.md) and
[REFERENCE_BENCHMARKS.md](REFERENCE_BENCHMARKS.md). The reference GraphGPS recipe uses
dense Transformer attention and LapPE, so the old Performer scaling description does
not apply to it. The 4096-node guard remains exclusive to the custom Graphormer model.

Scope: request 21, graph-size limits, observed scalability, and failure handling. The broader attached review is context; this report does not certify the full architecture/fairness audit in items 1–20. No model tuning was performed.

## Graphormer: what 4,096 means

**4,096 is a custom EfficientGraphBench memory-protection threshold. It is not an inherent Graphormer architectural limit, and it was not established by an OOM boundary experiment.** It was introduced in this project to avoid large dense allocations. The exact enforcement point is `src/efficientgraphbench/models/graphormer.py`, `prepare_graphormer()`, through `MAX_NODES = 4096`, before the SciPy adjacency/shortest-path construction. The runner invokes it after graph validation and before GPU transfer; the model also invokes it when preprocessed tensors are absent.

The original Microsoft implementation has configurable dataset/task limits: [GraphPredictionConfig.max_nodes](https://github.com/microsoft/Graphormer/blob/main/graphormer/tasks/graph_prediction.py) defaults to 128, while the [collator](https://github.com/microsoft/Graphormer/blob/main/graphormer/data/collator.py) accepts max_node and filters oversized graphs. These are pipeline settings, not evidence for a universal 4,096-node bound. This audit found no basis in those sources for calling 4,096 an original Graphormer restriction.

At N=4,096, one int64 N×N distance tensor is 128 MiB and one float32 four-head attention-bias tensor is 256 MiB. The float64 SciPy distance matrix is another 128 MiB on CPU. These are individual tensors, not total peaks: copies, gradients, layers, optimizer state, features and attention workspaces add memory. N=8,192 makes each quadratic term four times larger. Current preprocessing constructs dense all-pairs distances before clipping distances, so the 31-hop bucket does not reduce its quadratic storage.

### Could it exceed 4,096?

Yes, technically, after changing the project guard and validating resource use. Larger RAM/VRAM, lower precision and gradient checkpointing can extend feasible sizes. Memory-efficient exact attention can reduce attention workspace but the current dense distance and bias tensors still cost O(N²); they would also need blockwise/on-demand handling. Moving preprocessing off GPU does not remove CPU O(N²) storage.

Batching independent graphs limits each batch, but splitting one transductive graph into node mini-batches or subgraphs changes global attention and potentially shortest-path information. Sparse/local/linear attention can scale farther but is a changed Graphormer variant, not a transparent optimization of this dense full-graph benchmark. Any such experiment must record the changed protocol and structural encodings. This change keeps the existing guard; it now reports UNSUPPORTED_GRAPH_SIZE rather than OOM.

## Evidence: maximum successfully tested graph size

Values below are maxima among surviving saved successful records, not supported maxima or fresh stress tests. The project overwrites active results; deleted history cannot establish a maximum. Smoke runs are distinguished by their epoch counts. Stored edge counts count directed edge-index entries, not necessarily unique undirected relationships. All listed maxima used the NVIDIA GeForce RTX 4050 Laptop GPU (6 GiB), PyTorch 2.14.0+cu130.

| Model | Largest successful N / E | Dataset | Epochs in evidence | Run evidence |
|---|---:|---|---:|---|
| Graphormer | 2708 / 10556 | cora | 200 | `experiments/raw/results.jsonl`; `20260919T211158-cora-graphormer-seed42-33cddb53` |
| GraphGPS | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211536-pubmed-graphgps-seed42-d8b86132` |
| GCN | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211757-pubmed-gcn-seed42-bd38ddfa` |
| GraphSAGE | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211804-pubmed-graphsage-seed42-4456c36e` |
| GAT | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211816-pubmed-gat-seed42-c2296450` |
| SGFormer | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211827-pubmed-sgformer-seed42-a22cb1f4` |
| MLP | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211844-pubmed-mlp-seed42-3430d910` |
| SGC | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211846-pubmed-sgc-seed42-e28bdf11` |
| APPNP | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211912-pubmed-appnp-seed42-e474a3df` |
| GATv2 | 19717 / 88648 | pubmed | 200 | `experiments/raw/results.jsonl`; `20260919T211922-pubmed-gatv2-seed42-d99f8c87` |

## Per-model limits and scaling

All ten models train full-batch on one complete graph, including validation/test node features but using training labels only. There is no neighbor sampling, Cluster-GCN, graph partitioning, or mini-batch trainer. N = nodes, E = stored edges, d = hidden width, F = input width, C = classes, H = heads, K = propagation steps. Orders below hold width/depth fixed and describe model work; dense input features require O(NF) for every model. Sparse graphs can still have E approaching N². Training retains activations across layers.

| Model | Hard-coded node/edge ceiling | Memory / attention scaling | Observed real OOM boundary |
|---|---|---|---|
| Graphormer | N ≤ 4,096 (project safeguard); no separate E ceiling | Dense global attention: O(HN² + Nd) bias/activations, O(N²) distances; attention compute O(N²d). | Not measured |
| GraphGPS | No explicit N/E ceiling found | Global Performer attention plus local GraphSAGE: O(Nd + NHr + Ed), r fixed random-feature width. | Not measured |
| GCN | No explicit N/E ceiling found | Sparse local aggregation; O(Nd + Ed) activations/messages; no dense attention. | Not measured |
| GraphSAGE | No explicit N/E ceiling found | Sparse local aggregation; O(Nd + Ed); no dense attention. | Not measured |
| GAT | No explicit N/E ceiling found | Local edge attention; O(Nd + EH + Ed) intermediate storage. | Not measured |
| SGFormer | No explicit N/E ceiling found | Global linear attention plus local GNN: O(Nd + Ed), with width-dependent terms. | Not measured |
| MLP | No explicit N/E ceiling found | Feature-only forward, O(Nd); benchmark still loads/transfers all E edges. | Not measured |
| SGC | No explicit N/E ceiling found | Sparse K-step propagation, O(NF + EF) working tensors; no attention; cache disabled. | Not measured |
| APPNP | No explicit N/E ceiling found | MLP plus sparse K-step propagation; O(Nd + NC + EC); no attention; cache disabled. | Not measured |
| GATv2 | No explicit N/E ceiling found | Local edge attention; O(Nd + EH + Ed) intermediate storage. | Not measured |

No actual allocator-exhaustion threshold was established by available records. Three historical PubMed Graphormer entries (19,717 nodes, 88,648 edges) labeled oom contain the project-limit message; these are UNSUPPORTED_GRAPH_SIZE evidence, not measured OOM. Tests injecting torch.OutOfMemoryError validate handling only, not capacity. Permission/download failures and incomplete records are not size evidence.

### Other explicit and implicit constraints

- Graphormer clips degree encodings at 511 and reachable shortest-path distances at 31, with a separate unreachable bucket (32). These change representation resolution; they do not reject graphs at those degree/distance values. Duplicate edges are collapsed for Graphormer structural encodings.
- `datasets/tables.py:read_table`, `datasets/custom.py`, `datasets/prepare.py`, and `datasets/registry.py` materialize complete tables/graphs. CSV strings, pandas copies, dense float32 features, and validation/fingerprint buffers can exhaust host RAM before a model starts. No numeric row/edge quota is imposed. PDF extraction and XLSX reading also accumulate selected tables in memory.
- `datasets/viewer.py` serializes all rows for HTML viewing; selecting 12 feature columns is a display default, not a training graph-size limit. Large viewers can exceed browser memory.
- Every model transfers the full graph to the chosen device. Hidden width, input width, layer count, attention heads, graph density, optimizer state and available memory all matter; node count alone cannot predict OOM.
- GraphGPS pads a single graph for Performer attention but does not create a full N×N attention matrix. SGFormer uses linear global attention and a local GNN. Neither wrapper imposes an N/E cap. Their advertised scalability does not establish this installation’s tested capacity.

## Failure handling implemented

New records use `SUCCESS`, `OOM`, `UNSUPPORTED_GRAPH_SIZE`, `CONFIG_ERROR`, or `RUNTIME_ERROR`. Legacy ok/oom/error records remain readable; the known historical Graphormer 4,096 guard is recognized separately. A rejected size is never classified as an actual allocator OOM in new runs.

Failed runs save dataset/model/seed/configuration, node and edge counts when loading completed, hardware identity and available memory, failure_reason, and implementation_limit where applicable. Counts are explicitly null with a graph_size_note if configuration or dataset loading failed before sizes were known; they are never invented. Config errors cannot safely load an unknown/invalid dataset just to derive counts. Hardware availability is a snapshot, not reserved capacity.

CLI comparison and saved reports include failure rows with reasons and limits. Success rows have SUCCESS status; failed runs never contribute to accuracy/timing averages. A failed model no longer aborts the rest of the matrix. Invalid file syntax/CLI argument parsing remains a CLI error before a runnable experiment exists. Result-storage failures themselves must still surface because no reliable record can be saved. Old logs/checkpoints remain according to the existing overwrite policy.

## Findings and changes

| Severity | File / function | Finding | Action |
|---|---|---|---|
| High | models/graphormer.py:prepare_graphormer | Custom guard previously raised MemoryError, falsely implying measured exhaustion. | Added UnsupportedGraphSize carrying limit ownership and enforcement location. |
| High | benchmark/runner.py:run_benchmark | Validation/device errors occurred before records; runtime exceptions aborted the matrix. | Save CONFIG_ERROR; save and return runtime/size/OOM failures with graph and hardware metadata. |
| Medium | cli/main.py:compare; results/report.py:write_report | Failed models omitted from output tables. | Show explicit failure statuses/reasons and SUCCESS rows. |
| Medium | benchmark/runner.py:_run_validated | Cleanup after failed models was implicit. | Drop graph/model references, collect garbage, and release unused CUDA cache between runs. |
| Medium | profiling/runtime.py:MemoryProfiler | Graphormer structural preprocessing precedes memory profiling. CPU peak does not include the whole preprocessing peak; cost is timed separately. | Disclosed; a future profiling audit should extend coverage and separately record reserved memory without conflating scope. |
| Medium | models/graphormer.py | Guard is static, not tuned to device or input width/depth. | Keep conservative guard and explicit status; future configurable budgets require stress-test evidence. |

No capacity stress test or architecture rewrite was performed for this audit. Follow-up scalability work should sweep N/E/features/depth on isolated processes and report successful and failed allocations with RAM/VRAM snapshots, rather than infer a threshold from one successful dataset.

## Validation of this change

83 automated tests passed. A fresh one-epoch, one-seed PubMed GPU matrix in `experiments/graph-size-audit` produced nine SUCCESS records and one UNSUPPORTED_GRAPH_SIZE record with N=19,717, E=88,648 and the custom 4,096-node limit. This confirms continuation and metadata, not trained accuracy or an OOM boundary.
