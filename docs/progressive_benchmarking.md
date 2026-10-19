# Progressive Multi-Fidelity Benchmarking

Models first train on an initial graph, then **resume their weights, optimizer and scheduler** as nodes and edges are added. Every model/seed has its own checkpoint chain. This mode aims to reduce computation through validation-based pruning and measure adaptation to graph growth. It is **expanded transductive retraining**, not an online-learning algorithm and not equivalent to a full static benchmark.

The existing `Benchmark` and `ScalingBenchmark` defaults are unchanged. Scaling starts fresh at each graph size; progressive training continues across sizes.

## Python: small local run

```python
from efficientgraphbench import ProgressiveBenchmark

results = ProgressiveBenchmark(
    dataset="texas",
    models=["gcn", "graphsage", "gat", "gatv2", "appnp", "sgc", "mlp"],
    seeds=[1, 2, 3],
    device="auto",
    output_dir="experiments/progressive-texas",
    progressive={
        "progression_strategy": "deterministic_nested",
        "progression_seed": 42,
        "selection_metric": "val_f1_macro",
        "min_candidates": 2,
        "promotion_margin": 0.01,
        "uncertainty_aware": True,
        "recovery_threshold": 0.95,
        "stages": [
            {"name": "initial", "graph_fraction": 0.5,
             "additional_epochs": 10, "retain_fraction": 0.7},
            {"name": "growth", "graph_fraction": 0.75,
             "additional_epochs": 10, "retain_fraction": 0.7},
            {"name": "full", "graph_fraction": 1.0,
             "additional_epochs": 20},
        ],
    },
).run()

results.show_all()                   # Stage tables, adaptation, outcomes and compute accounting
results.show_all(open_browser=True)  # Browser report; Ctrl+P to save as PDF
results.summary()                    # DataFrame of final/last-observed outcomes
print(results.output_dir)            # New unique directory per experiment
```

Alternative entry point: `Benchmark(..., benchmark_mode="progressive_multi_fidelity", progressive={...})`. Default `benchmark_mode="full"` retains the existing runner. `ProgressiveConfig` and `FidelityStage` are also public imports. `runs=10` can replace `seeds`; do not supply both. Use a fixed source split (`split_index`, `split_seed`, `split_mode`); folds are not cycled across stages or seeds in this mode.

## Terminal

```powershell
.venv\Scripts\egbench progressive --config configs/progressive_texas.yaml
```

Linux / activated environment:

```bash
egbench progressive --config configs/progressive_texas.yaml
```

## Budgets and selection

`additional_epochs=30` always means **30 more epochs**, not a cumulative target. Continuation uses the last epoch, not a best-validation checkpoint, so optimizer and model state refer to the same update. Early stopping is disabled in v1; configuring it raises an error. A cosine scheduler, when selected, spans the sum of all stage budgets and resumes its state.

After every non-final stage, calculate mean validation score, sample standard deviation and successful seed count. Rank descending, breaking exact mean ties by model ID. The nominal retained count is `ceil(successful_candidates * retain_fraction)` or `retain_top_k`, floored by `min_candidates` and capped at the successful candidate count.

The **fixed nominal cutoff candidate** defines the boundary. Candidates outside the nominal set also advance if either:

1. `promotion_margin > 0` and their mean is at least `cutoff_mean - promotion_margin`; or
2. `uncertainty_aware=True` (default), both candidate and cutoff have multiple seeds, and `candidate_mean + candidate_sd >= cutoff_mean - cutoff_sd`.

These safeguards can retain more than top-K; the cutoff is not recursively moved. Mean ± sample SD is a descriptive variability band, **not a significance test or confidence interval**. With one seed, uncertainty protection is unavailable. Set `uncertainty_aware=False` and `promotion_margin=0` for strict nominal pruning. Failed/OOM candidates never advance, regardless of margin or minimum. One failed seed disqualifies the candidate. Decisions save ranks, cutoff statistics, dispersion, successful seed count and reasons. Final-stage candidates are completed, not pruned again.

Test labels never enter loss or promotion. Test metrics are evaluated only on the final graph, after adaptation, and saved separately. No weighted composite score is used.

## Graph progression and leakage boundaries

The default `auto` uses temporal ordering when `node_time` or `node_year` exists, otherwise deterministic nested prefixes. A single `progression_seed` defines a shared, label-blind node order across all models and training seeds. Its prefixes create induced subgraphs; original masks and node identities are preserved. No validation/test labels are used to stratify ordering. Small stages can lack classes; training-class counts are recorded. A stage lacking training or validation nodes fails clearly: increase the initial fraction. Labels/classes must use a fixed, declared output vocabulary.

Temporal mode currently accepts a `DatasetBundle` with:

- One finite `node_time` (or `node_year`) per node, in consistent units.
- One finite `edge_time` per edge.
- `temporal_static_features=True`: an explicit assertion that static feature values and labels are available at node arrival, and preprocessing uses no future data.

Edges are included only when both endpoints have arrived and the edge timestamp is at or before the stage cutoff. Timestamp ties are broken by original node ID and may be split between prefixes. The final node arrival must cover all edge timestamps; later edge-only growth and evolving feature/label snapshots need a future adapter. No dates are inferred. Missing temporal metadata fails instead of quietly claiming temporal validity. A deliberate `deterministic_nested` run on timestamped data is a **non-temporal simulation**.

The existing OGB loader does not preserve edge-availability timestamps. For `ogbn-arxiv`, auto progressive mode refuses to claim temporal support: explicitly choose `deterministic_nested`, or supply a verified temporal bundle. CSV imports currently retain the static graph contract; provide temporal tensors through `DatasetBundle`.

## Supported models

Progressive resume is enabled for the current in-process adapters: GCN, GraphSAGE, GAT, GATv2, APPNP, SGC, MLP and explicitly selected PyG SGFormer (`sgformer_pyg`). Existing library/reference implementations are reused unchanged. Fresh adapter objects at stage boundaries load full checkpoint state; graph caches are not reused.

Official SGFormer (`sgformer`, alias `sgformer_reference`) and GraphGPS (`graphgps`) now continue through their isolated, pinned reference workers. Small actual CPU integration tests verify saved/restored model and optimizer signatures, cumulative optimizer steps, upstream commits and stage lineage. `sgformer_pyg` remains a separate identity. Graphormer remains `progressive_resume_unsupported`.

GraphGPS uses the pinned Actor node-classification recipe: GCN+Transformer and LapPE (four eigenvectors), with RWSE disabled. Each stage recomputes upstream LapPE on its own topology before loading trainable state. Dense eigendecomposition and attention still limit scalability. Arbitrary alternative recipes are not exposed. Strict state loading rejects incompatible tensor shapes; no resizing or replacement architecture is used.

SGFormer keeps upstream parameter groups and its no-scheduler recipe. GraphGPS restores its upstream cosine-with-warmup scheduler using the summed stage budget. Both use an external interface wrapper only; upstream repositories remain unchanged. Configured generic controller optimizers do not override official recipe optimizers. Exact additional budgets disable reference early stopping in Progressive mode.

## Adaptation metrics

All adaptation scores use the configured **validation** metric:

- **Old-node score:** validation nodes already present in the previous stage.
- **New-node score:** validation nodes introduced in this stage.
- **Overall score:** all current validation nodes; used for promotion.
- **Old/new pre-training scores:** measured immediately after graph expansion and checkpoint restoration, before any update. The first stage has no transition reference.
- **Post-expansion delta:** old-node pre-training score minus previous-stage validation score. Positive means expansion helped; negative means degradation.
- **Old-node performance delta:** old-node post-training score minus previous-stage score. Positive means improvement.
- **Old-node forgetting:** `max(0, previous-stage score - old-node post-training score)`. Never negative.
- **Old-node adaptation delta:** old-node post-training minus old-node pre-training.
- **New-node adaptation gain:** new-node post-training minus new-node pre-training.
- **Old/new gap:** absolute difference between post-training old and new validation scores. Diagnostic only, never a selection objective.
- **Recovery:** target is `recovery_threshold * previous-stage overall validation score`, tested on those same old validation nodes after expansion. Status is `recovery_not_required` if already at target (zero epochs/seconds), `recovered` if reached during additional training, `not_recovered` if the budget ends below target, or `not_applicable` without a valid reference. `recovery_epochs` and `recovery_seconds` record the first crossing. Both stage and finalist recovery means include recovered transitions only. Stage tables cover their stage; finalist recovery statistics pool all transitions. Counts show transitions requiring recovery, first-hit recoveries, never-recovered transitions and final target satisfaction. `first_recovery_epoch`/`first_recovery_seconds` retain the first hit; `final_target_satisfied` separately exposes relapse. Not-required transitions retain zero cost in raw records.

All signed changes use **after minus before**. Reports translate older saved metric names without rewriting their evidence; missing old new-node pre-training scores remain unavailable. New records no longer use `expansion_shock` or `old_node_performance_drop`.

The full graph is visible during transductive inference, but losses only use train masks. This does not establish genuine online inductive deployment behavior. CUDA operations without deterministic kernels can still vary; deterministic settings and software hashes are recorded, not a promise of bitwise GPU equality.

## Saved evidence and timing

Each unique experiment directory contains:

- `graphs/source`, `graphs/node_order.pt`, `graphs/stage-N`: portable graph/split interchange plus deterministic original node IDs in result metadata.
- `checkpoints/<model>/<seed>/<stage>.pt`: model, optimizer, scheduler, Python/NumPy/Torch/CUDA random states, cumulative epoch, config, software and graph hashes, parent checksum.
- `progressive-results.json`: complete stage/seed history, lifecycle decisions, adaptation, final test metrics, hardware and source provenance.
- `progressive-summary.csv` and `progressive-report.html`: last-observed results with explicit full-graph reach status, per-stage decisions, failures and compute accounting.

Restore checks checksum, model/config, seed, source graph, parent stage/graph and software provenance before loading. Only locally created trusted checkpoints are used; arbitrary external pickle files are not accepted by the controller. This release resumes **between stages within one invocation**; automatic restart of an interrupted process is not yet exposed.

Per-stage training time includes gradient updates and recovery checks. Stage memory uses sampled process RSS and CUDA allocator peaks; additional memory is the peak above the stage-entry baseline, not a sum across stages. Reported GPU memory is not total device usage. Total training sums all model/seed/stage work; wall time additionally includes graph preparation, model restore, evaluation, checkpoint I/O, failed attempts and report overhead before the accounting snapshot.

## Verified Full-vs-Progressive comparison

Set `full_baseline="experiments/full/full-benchmark.json"` (or the full experiment directory) only when saved full results exist. The default reports `full_benchmark_baseline = unavailable`. No Full Benchmark is launched automatically. Unverified timing estimates and the old `full_benchmark_seconds` input are no longer accepted.

Compatibility requires exact candidate/seed coverage without duplicate attempts, all successful static runs, identical full source graph and split hashes, hardware/runtime details, focused execution fingerprint, software versions, implementation source/commit, model config, optimizer/scheduler settings, threads, split settings and full epoch budget equal to the sum of progressive budgets. Missing or mismatched evidence yields `unavailable` with a reason. This is conservative; it does not declare the two learning protocols equivalent.

A compatible baseline supplies the sum of `training_time_sec`. Wall-clock comparison additionally requires a saved measured full-matrix envelope:

```json
{
  "metadata": {
    "elapsed_seconds": 120.0,
    "elapsed_scope": "full_matrix",
    "run_ids": ["exact-run-id-1", "exact-run-id-2"]
  },
  "records": ["actual run-record objects go here"]
}
```

Full matrix execution now automatically writes `full-benchmark.json` with measured elapsed time and exact run IDs. The example structure is illustrative; do not invent times. Older Full results remain readable, but baselines without the focused execution fingerprint are rejected by the new controller with an explicit rerun instruction. Summing per-worker times is not accepted as total full-matrix elapsed time.

- `training_time_reduction_percent = 100 * (full_training - progressive_training) / full_training`
- `wall_clock_reduction_percent = 100 * (full_elapsed - progressive_elapsed) / full_elapsed`

Negative wall-clock reduction is labeled `progressive_overhead_exceeds_savings`. Zero is `no_wall_clock_change`. No unsupported savings values are estimated. Training time includes each protocol's validation checks, so it is practical training cost rather than pure optimizer-kernel time. Progressive elapsed time includes initial report generation; final accounting/report refresh is excluded and labeled.

## Report layout and protocol separation

The report begins with dataset nodes/edges/features/classes, progression, stage node counts, additional epochs, checkpoint policy, validation metric, seeds and candidate counts. **Full-graph finalists** have separate tables from **Early-eliminated candidates** and **Failed/skipped/unsupported candidates**. A model is a finalist only if all requested seeds completed the final graph successfully. Stage history aggregates per seed before displaying pre/post measurements, signed deltas, forgetting, gap, recovery states, training time and memory. No partial-graph score is presented as a full-graph result.

`summary()` remains a machine-readable combined history summary with a `Finalist` flag; the user-facing report separates the populations. Browser print uses dark portrait pages with narrow table groups, repeated headers and no forced section page breaks. Measurement notes and the metric glossary are together at the end. Raw JSON retains every attempt and checkpoint lineage.

Only Full Benchmark and Progressive Multi-Fidelity are supported protocols here. No audit, selection-regret or winner-retention mode is provided. Ordinary Full Benchmark behavior and its exhaustive model set remain unchanged.


## Refined measurement and compatibility contract

Full aggregation rejects progressive or mixed-protocol records, including legacy-looking records carrying stage/adaptation fields. Unambiguous old Full records remain readable. Candidate selection verifies the exact configured seed set; omitted and duplicate seeds cannot advance. Reports expose nominal, rank-, margin- and uncertainty-retained counts.

`optimization_seconds` excludes validation; `validation_seconds` covers recovery checks in Progressive and every-epoch validation in Full. Progressive also records pre/post-adaptation evaluation, test evaluation, checkpoint save/restore and graph build durations. Failed attempts retain elapsed and available partial phase timings. Optimization + evaluation + other execution reconciles to measured total elapsed. Other execution includes useful initialization, graph preparation, checkpoint I/O, subprocess startup, persistence and reporting; it is not necessarily waste. Diagnostic failed-attempt elapsed and training-phase totals overlap these categories and must not be added to them.

Full retains best-validation-accuracy weights and every-epoch validation. Progressive retains last-epoch weights and checks recovery each epoch until the first hit, plus pre/post stage evaluation. Both policies are recorded. Equal epoch counts do not imply equal graph exposure; observed differences cannot be attributed solely to graph growth. `training_time_sec` remains a legacy training-phase duration; the new optimization comparison uses a matching scope. No automatic Full rerun occurs.

Execution fingerprints include graph/splits, recipes/seeds, pinned implementation commits, execution source hashes, Python/Torch/PyG/NumPy/SciPy/CUDA/cuDNN versions, physical hardware and timing methodology. Report CSS, charts and documentation are excluded. Raw full source snapshots remain available for provenance. A compatible baseline must cover every selected model and seed successfully.

Automatic recovery of a benchmark after the controller itself is forcibly terminated remains unsupported. Owned worker timeouts are recorded as failed elapsed time, but a killed worker cannot return its unfinished internal phase counters.
