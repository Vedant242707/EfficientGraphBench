# Complete implementation review: all 20 sections

**Historical pre-migration review.** Findings below describe the code at that audit.
The subsequent reference-runner upgrade is tracked in
[REFERENCE_MIGRATION_STATUS.md](REFERENCE_MIGRATION_STATUS.md).

All 20 sections of the attached review were examined. **Reviewed does not mean passed.** The current benchmark is a functioning common-configuration node-classification experiment, not yet a research-grade comparison of the original published architectures. Architecture and training code were not changed during this audit; no test-driven tuning was performed.

Evidence: [audit snapshot](audits/review20/evidence.json), [model classes and source hashes](audits/review20/model_provenance.json), and the diagnostic scripts/results under `audits/review20/`. Environment: PyTorch 2.14.0+cu130; PyG 2.8.0.post1; NumPy 2.4.6; SciPy 1.17.1; pandas 3.0.5; RTX 4050 Laptop GPU, 6 GiB. Source hashes identify this audit snapshot, not necessarily the code used for historical runs. The separate [scalability review](GRAPH_SIZE_SCALABILITY_REVIEW.md) covers section 21.

## A. Correct and research-ready components

These mechanisms are sound within their stated scope, not a blanket certification:

- A shared loader supplies the same data and masks to each model; the graph fingerprint includes all three masks.
- Training loss uses training labels. Validation chooses a checkpoint; the selected checkpoint supplies the test score. Test labels do not choose epochs.
- **All 41 surviving 200-epoch Cora checkpoints reproduced their saved test accuracy exactly** when reloaded on the current GPU. This includes old and new experiment groups, not 41 independent trials.
- Seed setup covers Python, NumPy, PyTorch and CUDA; saved experiments use distinct seeds 42, 43, 44 where records survive.
- Inference uses evaluation mode, inference_mode, warm-up, synchronization and repeated timings.
- Input validation checks numeric finiteness, IDs, edge bounds, label ranges, disjoint/nonempty masks and training-class coverage. NPZ disallows pickle.
- Current failure handling distinguishes the five requested statuses and allows the matrix to continue after returned model failures.
- **83 automated tests passed** in this review. These tests support pipeline behavior, not equivalence to original research repositories.

## B. All 20 checklist sections

### 1. Model implementation correctness — reviewed; material differences

| Model | Actual implementation | Interpretation |
|---|---|---|
| GCN | PyG nn.models.GCN / GCNConv | Official library model; common settings, not a paper-specific reproduction. |
| GraphSAGE | PyG nn.models.GraphSAGE / SAGEConv | Official library model with mean aggregation; full-batch rather than original sampling experiment. |
| GAT | PyG nn.models.GAT / GATConv | Official library model. |
| GATv2 | PyG GAT with v2=True / GATv2Conv | Correct operator selection, verified by tests. |
| SGC | PyG SGConv | Single linear classifier after K propagation steps; cache disabled. |
| APPNP | Local APPNPModel with FeatureMLP + PyG APPNP | Custom predictor using official propagation operator. |
| MLP | Local FeatureMLP | Feature-only control, intentionally ignores edges. |
| SGFormer | PyG nn.models.SGFormer | Library variant; not the authors' repository imported directly. |
| GraphGPS | Local wrapper over GPSConv, SAGEConv and Performer | Simplified/adapted GPS recipe. |
| Graphormer | Local models/graphormer.py using torch MultiheadAttention | Custom node-classification adaptation, not Microsoft's model or pretrained weights. |

No OpenGT implementation is used. `models/registry.py:build_model` and the provenance JSON identify exact installed classes and hashes.

**Confirmed SGFormer difference:** installed `SGFormerAttention.forward` normalizes Q/K per node/head along the feature dimension. The authors' [full_attention_conv](https://github.com/qitianwu/SGFormer/blob/main/medium/ours.py) uses tensor-wide norms. These produce different attention magnitudes. PyG always applies ReLU after attention/residual/norm, while the inspected author implementation makes it optional. PyG also supplies its own residual GCN branch. These are material variant differences; their individual effects on Cora accuracy have not been isolated.

| Component | SGFormer here | GraphGPS here | Graphormer here |
|---|---|---|---|
| Attention | Global linear Q/K/V plus local GCN | Global Performer plus local GraphSAGE | Dense global softmax with per-head shortest-path bias |
| Normalization | torch LayerNorm globally; BatchNorm locally | PyG LayerNorm, default mode=graph | torch per-node LayerNorm |
| Residuals | Averaged global residual; library GNN residual path | Local/global residuals, branch sum, FFN residual | Pre-norm attention and FFN residuals |
| FFN | Input projections and classifier; no standard FFN per attention block | GPSConv includes FFN | 4× width GELU FFN per block |
| Dropout | Default 0.5 in both branches | Wrapper/block and Performer output dropout | Attention, FFN and residual dropout |
| Positional encodings | None expected by this API | None; omitted from full recipe | No spectral PE |
| Structural encodings | Local graph branch | Local edges only; no RWSE/LapPE | In/out-degree and shortest-path embeddings |
| Graph token | None | None; node head | Omitted from original graph-level design |
| Masks | Padding mask for one full graph | Padding mask for one full graph | Finite structural bias; disconnected pairs remain attendable |
| Edge encodings | Untyped GCN connectivity | Untyped SAGE connectivity | Untyped shortest paths, no molecular edge/path-type encoder |
| Outputs | Per-node log-probabilities | Per-node logits | Per-node logits |

GraphGPS's `layer_norm` is graph-wise PyG normalization, not necessarily torch per-node LayerNorm. The [official GPS framework](https://github.com/rampasek/GraphGPS) supports positional/structural encodings and multiple operator choices. This implementation fixes one combination and omits those encodings; it is not a faithful reproduction of a specific tuned configuration.

Graphormer changes the original input embedding, head, token, edge encoding, initialization and training regime. Its torch Embedding initialization differs from the small-scale initialization in the inspected [Microsoft layers](https://github.com/microsoft/Graphormer/blob/main/graphormer/modules/graphormer_layers.py). A seed-42 Cora diagnostic measured initial projected-feature RMS 0.01583 versus summed degree-embedding RMS 1.38472, approximately 87× larger. This is a measured initialization imbalance, not proof that it alone causes the accuracy gap.

### 2. Transformer input handling — reviewed; valid shapes, adapted semantics

All models receive x and edge_index with the expected dimensions. SGFormer and GraphGPS receive an all-zero batch vector for one graph. Graphormer receives degree vectors and an N×N spatial tensor derived from edge_index. Validation/test labels are not model inputs.

Cora and PubMed features are row-normalized for every model. This common policy is not necessarily appropriate for every reference configuration. Graphormer uses a linear encoder for continuous features, omits graph token/typed edges, clips degrees at 511 and distances at 31, and uses a separate unreachable bucket. GraphGPS has no additional PE/SE. SGFormer does not require shortest paths or a missing positional encoder.

Graphormer's cached structural tensors are correct for the static runner. A caller changing edges after preprocessing would need to invalidate them; the cache is not bound to a topology fingerprint.

### 3. Dataset splits — reviewed; mechanism passes, custom-file access limited

Freshly reloaded data:

| Dataset | Nodes | Edges | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 140 | 500 | 1,000 |
| PubMed | 19,717 | 88,648 | 60 | 500 | 1,000 |

`datasets/registry.py:load_dataset` uses Planetoid public masks. Other unassigned nodes still participate in the transductive graph. Custom CSV masks are provided or generated once by prepare; no model creates a split. Saved groups have matching fingerprints/counts across models. Complete fingerprints and separate train/validation/test SHA-256 hashes are recorded in audit evidence.

The current `data/my_dataset/nodes.csv` returned Windows Permission denied. Its pipeline was checked through source, fixture tests and saved metadata, but **fresh validation of that specific file remains unverified**. The normal runner saves counts and a combined fingerprint; it does not log all counts at startup or store a separate per-mask hash.

### 4. Training protocol — reviewed; identical settings are not identical suitability

`training/trainer.py:train` uses Adam, lr 0.01, weight decay 0.0005, 200 epochs; model defaults are width 64, nominal depth 2, dropout 0.5. Cross-entropy uses train_mask only. No scheduler, warm-up, clipping, early stopping, architecture-specific optimizer groups or hyperparameter search exists.

Nominal depth and width mean different capacities across architectures. SGC has one classifier and K propagation steps; Transformer blocks add FFNs/projections; SGFormer has separate global and local branches. Some models ignore common fields.

The authors' [SGFormer Cora command](https://github.com/qitianwu/SGFormer/blob/main/medium/run.sh) uses four GNN layers, graph blend 0.8, global dropout 0.2, separate regularization, unnormalized features and a class-random split. The current wrapper uses one global layer/head, two GNN layers, blend 0.5 and dropout 0.5. Do not silently replace the common public split to match that command. [Graphormer training examples](https://github.com/microsoft/Graphormer/blob/main/docs/Quick-Start.rst) and [an official GPS configuration](https://github.com/rampasek/GraphGPS/blob/main/configs/GPS/zinc-GPS%2BRWSE.yaml) also use different schedules; they concern different tasks and are not automatically Cora-optimal recipes.

Under-tuning is plausible but not causally quantified. Equal validation-search budgets and declared per-model configurations are needed for stronger claims.

### 5. Selection and early stopping — reviewed; fixed-epoch selection passes

Each epoch evaluates val_mask in eval/inference mode. Strictly better validation accuracy stores a CPU clone of the complete state_dict; ties retain the earliest best epoch. After all epochs, that state is saved and loaded, then tested. There is no early stopping. All 41 surviving 200-epoch Cora checkpoints reproduced their test metrics.

Logs contain epoch 1, every 25 and final epoch, not structured full loss/validation curves. This limits plateau and overfitting analysis. SGFormer's log_softmax output passed to cross_entropy is mathematically equivalent to NLLLoss apart from rounding; an audit diagnostic confirmed agreement. It is not an established explanation for its low score.

### 6. Seeds and reproducibility — reviewed; partial guarantee

`training/seed.py` seeds Python, NumPy, torch and CUDA, sets CUBLAS_WORKSPACE_CONFIG, disables cuDNN benchmark and requests determinism. `warn_only=True` permits nondeterministic operators to execute, so strict cross-device/version reproducibility is not guaranteed. No Python-hash-dependent model ordering was found; PYTHONHASHSEED is not set.

Saved three-run experiments use 42/43/44 with fixed data fingerprints. The active 200-epoch Cora group now contains **29 successes**, because GCN seed 42 was replaced by a CONFIG_ERROR from a validation invocation in the preceding audit. The old `experiments/cora-gpu` archive still contains that seed's matching score. This exposed a retention bug: invalid invocations can overwrite valid metrics. Tests now direct invalid invocations to temporary folders, but production overwrite semantics still need correction. Do not report three GCN runs for the currently surviving two-run active group.

### 7. GPU memory — reviewed; suspicious values reproduced, scope incomplete

`profiling/runtime.py:MemoryProfiler` resets peaks on entry and reads `torch.cuda.max_memory_allocated()` on exit. GPU MiB is the across-run mean of peak allocated bytes / 2²⁰. It is not reserved memory, NVML usage or total GPU memory. Peak reset does not clear live tensors. Current cleanup drops graph/model references, runs GC and empty_cache, but library allocations may remain. CPU RSS sampling every 10 ms can miss short peaks.

Fresh-process Cora probes used seed 42, two training epochs, 10 warm-ups and 50 inference repeats. Each started with zero torch allocated/reserved bytes:

| Model | Peak allocated MiB | Peak reserved MiB | Allocated after cleanup MiB |
|---|---:|---:|---:|
| GraphSAGE | 169.209 | 186 | 64 |
| SGC | 254.715 | 262 | 64 |
| Graphormer | 854.455 | 918 | 64 |
| SGFormer | 118.212 | 142 | 64 |

These reproduce the historical allocated-memory values without a preceding model. They are not simply inter-model accumulation. The owner of the 64-MiB retained allocation was not traced; attributing it to a particular library would be speculation. SGC propagates 1,433-dimensional features before classification; SAGE also handles wide input messages. Parameters therefore do not predict activation memory. Graphormer additionally has dense attention/bias tensors. Operator-level allocation tracing remains recommended.

The reserved-memory fields above were collected only for diagnostics; normal benchmark records still lack them. `memory-*.json` contains full probe outputs.

### 8. Training time — reviewed; consistent boundary, incomplete breakdown

The timer starts after optimizer construction and includes epoch optimization, validation, synchronization, best-state CPU copies and logging. It excludes load/preprocessing, model initialization, GPU transfer, optimizer construction, disk checkpoint save, final reload and final test evaluation. Optimization and validation subtimers are saved separately.

This is not end-to-end runtime. Add initialization, transfer, checkpoint I/O, test-evaluation and total-runtime fields without retrospectively redefining training time.

### 9. Inference latency — reviewed; mechanism passes within warm full-graph scope

`measure_latency` uses eval/inference_mode, 10 warm-ups, synchronization before and after each interval, and 50 repeats. It saves median as inference_ms and mean as inference_mean_ms. The table averages per-run medians across seeds. It measures full-graph warm forward time, not per-node or cold application latency.

Dataset loading, transfer and Graphormer shortest paths are excluded. SGC/APPNP recompute propagation with caching disabled; Graphormer reuses structural preprocessing. These boundaries must be disclosed. Samples, SD, tail quantiles and power/clock state are not saved. Small models can be dominated by Python launch/synchronization overhead; sub-millisecond times are possible but not an end-to-end speed guarantee. Independent CUDA-event/cold-latency validation was not performed.

### 10. Parameter count — reviewed; saved but absent from comparison

Runner saves total num_parameters; inspector counts requires_grad only. They agree for all current default models, but would diverge if parameters were frozen.

Default Cora trainable counts: GCN 92,231; GraphSAGE 184,391; GAT 93,759; GATv2 187,355; MLP 92,231; SGC 10,038; APPNP 92,231; SGFormer 205,255; GraphGPS 175,559; Graphormer 257,995. These were independently counted in the provenance artifact.

Add explicit trainable_parameters/total_parameters fields and the requested comparison column. That addition is recommended, not already implemented by this audit.

### 11. Preprocessing — reviewed; timing exists, profiling gaps remain

preprocessing_time_sec covers dataset loading, validation, fingerprinting and Graphormer structural encoding. It can include downloads/cold cache effects; it conflates I/O and transforms. Each model reloads data, so the first may pay costs later models avoid. Graphormer recomputes structural encoding per run and reuses it within that run.

Memory profiling begins afterwards, missing transient CPU parser/shortest-path peaks. If preprocessing raises, elapsed preprocessing time is not assigned and CPU memory can report zero despite earlier work. Add phase timing in finally and distinguish raw I/O, common transforms, model-specific preprocessing, initialization and transfer. This is a measurement gap, not evidence that preprocessing was secretly counted as training.

### 12. Provenance — reviewed; insufficient per-run detail

Records contain names/family/attention, some adaptation notes, configuration and torch/PyG versions. They do not uniformly store source type, repository URL, paper, source commit/hash or detailed modifications. Lockfiles exist but their identity is not embedded. No workspace git revision was available; the audit saved SHA-256 hashes instead. Audit provenance does not prove which exact historical source produced old records.

Original associations: GCN/Kipf–Welling, GraphSAGE/Hamilton et al., GAT/Veličković et al., GATv2/Brody et al., SGC/Wu et al., APPNP/Klicpera et al., SGFormer/Qitian Wu et al., GraphGPS/Rampášek et al., Graphormer/Ying et al. MLP is a generic control. Sources: [PyG operators](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html), [SGFormer authors](https://github.com/qitianwu/SGFormer), [GraphGPS authors](https://github.com/rampasek/GraphGPS), [Graphormer authors](https://github.com/microsoft/Graphormer). Future results need a uniform versioned provenance object and compatibility signature.

### 13. Task compatibility — reviewed; adaptations must be identified

Every forward returns N×C node predictions. The GNNs, propagation methods and SGFormer naturally support node classification; MLP is a valid control. GraphGPS and Graphormer use node heads and a full transductive graph instead of original graph-level configurations. Valid output shapes do not establish optimal adaptation or paper fidelity.

Cora executes with all wrappers. PubMed exceeds this project's 4,096-node Graphormer safeguard, not an inherent architectural bound. Graph regression/classification, multilabel, weighted/typed graphs, temporal/inductive evaluation and neighborhood mini-batching are outside current scope.

### 14. Fairness — reviewed; common setting, not general architecture ranking

Within groups, hardware, data/masks, nominal settings and seeds are controlled. The comparison hash includes library versions, configuration, content and preprocessing, but not implementation hash, Graphormer constants or every library default. Changed source can share an old group. Irrelevant fields/path spelling can also split equivalent comparisons. Per-model tuned settings currently form different groups rather than one controlled experiment manifest.

Predeclare data and masks; separate common-setting baselines from reference/tuned variants; allocate equal validation-search trials or wall-time budgets; select only on validation; freeze settings and evaluate declared fresh seeds. Report capacity, time, memory, failures and actual completed seeds. Preserve an immutable manifest while retaining the user's latest-only display. Never treat equal width/depth as equal capacity or tune on test accuracy.

### 15. Failures — reviewed; recent improvements, remaining setup/retention gaps

Section-21 changes save five distinct statuses, reasons, graph sizes where known, hardware snapshots and implementation limits. Matrix continues after returned failures. Early configuration/download failures can have null graph size with an explanation. Actual MemoryError/torch OutOfMemoryError is distinct from the custom guard; not every OS/driver allocation failure necessarily uses those exception types.

Some setup (threads, seeds, filesystem/logger initialization) is still outside the main try/finally. Device metadata collection can itself fail. CLI parsing and result-store failures can prevent a runnable/savable experiment. An invalid single run can replace a success, as section 6 demonstrates. These are outstanding robustness issues, not evidence of unrecorded successful predictions.

### 16. Custom CSV pipeline — reviewed; untyped/unweighted only

| Concern | Current behavior |
|---|---|
| IDs/endpoints | Arbitrary IDs map to row indices; missing, duplicate/empty IDs and invalid references rejected. |
| source_node/target_node headings | Require explicit --source/--target during prepare; canonical columns are source,target. |
| Duplicate edges | Preserved by CSV loader; operators can handle them differently. Graphormer structural preprocessing deduplicates. |
| Directedness | Preserved as supplied; no automatic reverse edges. |
| Self-loops | Preserved on import; GCN/GAT/APPNP/SGC have operator-specific loop/normalization behavior, SAGE a root path. |
| weight/relation_type | Not used. Prepare selects endpoints only for trained graph; original table retains extra columns. Direct custom loader also ignores extras. |
| Features | Numeric selection becomes feature_i; nonfinite values rejected. Custom features are not automatically row-normalized. |
| Labels | String labels mapped to integers; blanks map to -1 and cannot enter train/val/test masks. |
| Splits | Provided masks retained or fixed per-class generation requiring ≥3 labeled nodes/class. |

Silently dropping weights/types is high severity if users expect the richer graph in the review request. Reject unsupported semantics or explicitly acknowledge the unweighted/untyped projection. Persist directedness, duplicate and loop policy rather than blindly symmetrizing/deduplicating. Automatic numeric-feature selection can include alternate IDs/outcomes or leakage; tensor validation does not detect semantic leakage or near-duplicate connected rows. Current real custom-file access limitation remains; fixture tests exercise the pipeline.

### 17. Result validation and accuracy gap — reviewed; causal attribution incomplete

Saved results support the stated Cora pattern for the current implementations. All surviving 200-epoch test metrics reproduced from checkpoints. Archived GCN three-seed results support approximately 0.8220; the current two surviving active GCN rows alone do not establish that three-run mean.

Best-validation epochs: GraphGPS 10/14/13, Graphormer 189/181/148, SGFormer 91/178/164. Early GPS selection and late Graphormer/SGFormer selection motivate curve/regularization/budget investigation, but do not prove overfitting or insufficient epochs.

| Explanation | Evidence |
|---|---|
| Wrong checkpoint | Not supported by 41 reload checks. |
| Different Cora masks | Not supported by shared loader and matching fingerprints/counts. |
| SGFormer log_softmax loss | Equivalent objective; not an established bug. |
| Reference/architecture mismatch | Confirmed adaptations and SGFormer normalization difference. |
| Unsuitable shared settings | Differences confirmed; causal impact unmeasured. |
| Graphormer initialization scale | Imbalance measured; training effect needs controlled ablation. |
| Missing GPS encoding | Confirmed omission; accuracy contribution unmeasured. |
| Universal Transformer inferiority | Not supported. |

No full original-repository reproduction or validation-only tuning study was performed. First pin and parity-test the reference implementations, then vary one factor at a time under a declared validation budget. Do not optimize blindly to match expected test scores.

### 18. Output — reviewed; requested columns incomplete

| Requested field | Saved now | Displayed now |
|---|---|---|
| Model/family | Name plus metadata.family | Name, terminal family order; no family column |
| Accuracy mean/std | Per-run accuracy | Mean/sample SD |
| Trainable parameters | Total num_parameters, currently equal | Missing |
| Training time | Yes and subtimers | Yes |
| Mean/median latency | Both | Mean across per-run medians only |
| Peak allocated memory | peak_gpu_memory_mb | Yes |
| Peak reserved memory | No | No |
| Preprocessing time | Completed preprocessing only | Missing |
| Seed count | Seed per record | Run count, not expected/completed unique-seed validation |
| Status | Yes | SUCCESS and separate failure rows |

These missing runtime-output additions are recommended changes, not completed by this review. The terminal has the requested family separator; Markdown still orders model groups alphabetically. Existing metrics were not silently redefined.

### 19. Metadata — reviewed; partial

Present: dataset/model/seed/config, hardware/device, versions, platform, task/protocol, counts/fingerprint, preprocessing, optimizer/batch type, result/config/log/checkpoint paths. Recent records add CPU and available RAM/VRAM snapshot; older schemas differ. Time is embedded in normal run IDs/logs, not uniform started_at/completed_at fields; config-error IDs lack timestamp encoding.

Missing/inconsistent: driver/device UUID, precision/TF32 settings, determinism warnings, implementation commit/hash, complete dependency manifest, paper/source attribution and modifications. Audit artifacts provide a snapshot, not automatic provenance for future runs.

### 20. Final review — complete

This file supplies A/B/C/D, all 20 dispositions, evidence, severity and recommended targets. Checked is not certified. Fresh access to the locked custom file, upstream parity and controlled tuning remain unverified; this is stated rather than inferred from passing tests.

## C. Potential bugs or unfair comparisons

No critical label-leakage or checkpoint-selection defect was demonstrated in the inspected Cora path. High-severity issues block stronger scientific claims.

| ID | Severity | File/function | Problem | Recommended fix |
|---|---|---|---|---|
| R1 | High | models/registry.py:build_model; installed SGFormerAttention.forward | Library normalization differs from author attention. | Pin reference, add golden-output/gradient tests, label variants explicitly. |
| R2 | High | models/graphormer.py:Graphormer/GraphormerBlock | Custom adaptation/initialization materially differs. | Validate against reference and run validation-only initialization/encoding ablations. |
| R3 | High | models/transformers.py:GraphGPS | No PE/SE, fixed local/global/norm choices. | Declare simplified variant; add separately versioned reference configuration. |
| R4 | High | config.py; training/trainer.py:train | Untuned shared settings disadvantage some architectures. | Equal-budget per-model validation search and recorded selection policy. |
| R5 | High | datasets/prepare.py:prepare_csv; custom.py:_load_csv | Weights/types discarded; edge semantics differ. | Explicit supported-schema validation and canonical edge-policy metadata. |
| R6 | High | results/writer.py:write_result/replace_results | Invalid calls replace successes; partial batch erases previous active results. | Stage attempts and promote a completed latest manifest. |
| R7 | High | benchmark/runner.py:_run_validated | Comparison signature lacks source/variant identity. | Add code/model/preprocessing signatures and compatibility policy. |
| R8 | Medium | profiling/runtime.py:MemoryProfiler; runner boundary | Reserved memory and preprocessing peaks omitted. | Phase-aware allocated/reserved/RSS metrics and optional subprocess isolation. |
| R9 | Medium | training/trainer.py:train | Missing phase timers and structured full curves. | Record all phases, train/val curves; preserve metric version definitions. |
| R10 | Medium | cli/main.py:compare; results/report.py | Missing requested columns and seed completeness. | Share formatter; include parameters, latency mean/median, preprocessing, reserved memory. |
| R11 | Medium | training/seed.py; runner metadata | Warn-only determinism, incomplete environment identity. | Record effective settings and warnings; offer explicit strict mode. |
| R12 | Medium | runner setup/config-error path | Startup/hardware errors can escape structured records. | Minimal safe record first and guarded collection/failure phases. |
| R13 | Medium | models/graphormer.py:forward | Mutable graph can retain stale encodings. | Bind cache to immutable topology/fingerprint or invalidate. |
| R14 | Medium | datasets/viewer.py; registry metadata | Generic feature/label IDs lack original dictionaries. | Persist available vocabulary/category mappings, mark unknown semantics. |
| R15 | Low | report.py and cli/main.py | Different model ordering; wrapped failure details. | Shared display grouping and full report details. |

## D. Exact recommended change sequence

1. **Protect evidence:** in results/writer.py retain attempts by run_id/batch_id and a latest-completed pointer. CONFIG_ERROR must not displace a successful metric. Stage batch promotion; separately atomic JSONL/CSV writes are not one transaction. Preserve latest-only UI without sacrificing immutable manifests.
2. **Provenance before model changes:** in models/registry.py add source_type, repository/paper, version/commit, modified flag and changes; hash local adapters/preprocessing. Persist compatible signatures in runner.py. Do not retrofit new provenance onto historical runs as fact.
3. **Reference parity:** test identical inputs/weights against pinned SGFormer normalization/attention; establish explicitly named Graphormer/GraphGPS adapters and golden tests. Existing broad model names should not hide changed implementations.
4. **Instrumentation without optimization changes:** extend MemoryProfiler for allocated/reserved/start/end/RSS by phase. Record load/transforms/structural preprocessing/init/transfer/train/checkpoint/test/latency and partial failure timings. Add trainable_parameters and total_parameters.
5. **Complete output:** share CLI/report grouping; add requested fields and expected/completed seeds. Preserve GNN/baseline then Transformer grouping. One-run SD=0 should not imply stability.
6. **Graph semantics:** validate weight/type support and declare duplicate/direction/loop policy before model-specific processing. Add leakage diagnostics without silently changing graph semantics. Recheck the custom file when readable.
7. **Controlled experiments:** expose SGFormer branch blend/dropout/depth/regularization, GPS encoding/operator/norm, and Graphormer initialization/encoding choices. Use fixed masks and equal validation budgets; freeze settings before final test seeds. Keep reference/tuned and common-setting experiments scientifically distinct.
8. **Reproducibility:** explicit timestamps, driver/device/precision/determinism metadata, dependency/code manifests and optional per-run process isolation. Do not infer universal capacity from one successful graph.

These are recommendations, not claims that all fixes are already implemented. The saved scores describe the current implementations; they do not establish general GNN superiority over graph Transformers.
