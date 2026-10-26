# Execution Analysis: validation and delivery record

Branch: `codex/execution-analysis`. Main is intentionally unchanged.

This is the initial delivery record. The [subsequent execution review](execution_analysis_review.md)
records successful real CUDA kernel capture on the current local installation,
new navigation, memory sampling and portable trace validation. It supersedes the
initial no-kernel limitation below only for the newly tested environment/runs.

## Phase results

| Phase | Delivered | Verification |
|---|---|---|
| 1. Inspection | Execution paths, confirmed timing losses, GraphGPS configuration and measurement contract | 12 existing refinement tests, including both installed official workers |
| 2. Failure persistence | Shared monotonic events, nested identities, incremental journals, worker attempt recovery | 15 event/refinement tests; real official CPU continuation |
| 3. Accounting | Exclusive interval partition, failure categories, independent clock domains, explicit unattributed work | Reconciliation tests; final event/report batch: 9 passed |
| 4. Epoch/stage timing | Epoch events, existing synchronized phase boundaries, checkpoint and selection events | Refinement regressions and selective-stage tests |
| 5. Detailed profiling | Selected epochs/models/seeds/stages, CPU operators, CUDA-event intervals, trace export | 23 combined event/profile/refinement tests; real CPU and CUDA GCN, official SGFormer and official GraphGPS |
| 6. Explorer | Overview, Epoch Breakdown, GPU Kernels, exports/reopening, full-report link, compact printable summary | Headless Edge interaction checks; zero JavaScript errors; screenshots inspected; PDF rendered with Poppler |
| 7. Compatibility | Wheel assets, documented limits, reproducibility and overhead evidence | Full suite: 219 passed; subsequent focused batch: 18 passed; selected-stage batch: 2 passed |

Counts overlap; they must not be added as distinct tests. The later focused batch
includes newly added interruption, injected OOM, actual child-process crash and
CPU weight-equality checks. Lint passes. PyTorch emits its existing TorchScript
deprecation warning; this is not a benchmark failure.

The first wheel build hit an existing OneDrive lock on `dist/*.whl`. A fresh
temporary source copy built successfully without dependency downloads; the wheel
contains the HTML viewer and instrumentation modules. No old result directory was
deleted to work around that lock.

## What was actually tested

- Native GCN and pinned official SGFormer/GraphGPS: 40-node generated validation
  graph, two epochs, one seed, on CPU and the local RTX 4050 Laptop GPU. These are
  small **real executions on synthetic graph input**, not published benchmark
  accuracy results and not evidence of large-graph feasibility.
- Real CPU profiler traces and positive CUDA-event durations were exported.
  **All three local CUDA exporters lacked kernel events.** Tests first exposed
  this and then verified explicit unavailable-capability reporting. CUDA kernel
  collection on Linux/CUPTI remains unverified, not declared functional.
- Official SGFormer/GraphGPS two-stage continuation retained parent checkpoint
  checksums, model/optimizer signatures, additional epoch counts and original
  implementation classes. Selected-stage profiling additionally passed for GCN
  and official GraphGPS: only epoch 2 of the requested final stage was traced.
- Profiling off/lightweight/detailed produced bitwise identical CPU GCN checkpoint
  tensors on the controlled test. This does not assert universal CUDA bitwise
  determinism.
- Injected `torch.OutOfMemoryError` and `KeyboardInterrupt` after one completed
  epoch preserved the first epoch, failed phase, attempted duration and saved
  failure evidence. OOM skipped the next seed without duplicating its events.
  These were simulated failures, not a large GPU exhaustion experiment.
- A real subprocess exited before writing its result. The parent retained elapsed
  time, a completed child preprocessing event and an incomplete root with unknown
  duration. It did not assign zero training time to unknown work.
- Saved exports reopened without training. Main Full report generation with real
  execution records remained functional. Browser tabs/filter/zoom/pan and PDF
  printing were exercised, rather than inferred from HTML generation alone.

## Confirmed GraphGPS diagnosis

The inspected pinned recipe uses two GCN+Transformer layers, width 64, four heads
and LapPE with four eigenvectors. The global Transformer branch uses dense global
attention; its attention storage can scale quadratically with node count. The
adapter invokes upstream dense LapPE eigendecomposition before transfer.

The new `structural_preprocessing` event actually ran in both CPU/CUDA reference
tests. Parent worker-attempt and subprocess spans retain time even if child output
is absent. A failed official run now leaves completed phase events, the failing
phase, and available memory peaks. Neither architecture nor graph is shrunk.

This does **not** establish which allocation caused the earlier college OOM, which
graph fraction peaked, or how the historical 875/2456-second remainders divide.
Those require a new measured college run. No historical measurements were edited
or retrospectively invented.

## Measured overhead

Reproduce with `python examples/measure_execution_overhead.py`.
Raw measurements and source hashes: [execution_overhead.json](execution_overhead.json).

GCN, 500 nodes, 2,000 edges, 16 features, 50 epochs, seed 7; three order-rotated
repeats after warm-up. Detailed profiling captured epoch 10 only. Values below are
median observations, not a statistical confidence bound or a large-graph forecast.

| Device / instrumentation | Total wall s | Difference from off | Training s | Difference from off |
|---|---:|---:|---:|---:|
| CPU / off | 1.9223 | — | 0.8257 | — |
| CPU / lightweight | 2.0548 | +6.9% | 0.8698 | +5.3% |
| CPU / detailed | 2.1916 | +14.0% | 0.9905 | +20.0% |
| CUDA / off | 2.9510 | — | 1.7368 | — |
| CUDA / lightweight | 3.3570 | +13.8% | 1.9711 | +13.5% |
| CUDA / detailed | 3.3497 | +13.5% | 2.0315 | +17.0% |

GPU peak allocated memory was 65.727 MiB in all three modes on this workload.
Process RSS observations are saved, but same-process RSS is not an incremental
profiler-memory measurement. Differences include run-to-run noise: detailed total
wall time being slightly lower than lightweight here does not prove it is cheaper.
No correction factor is subtracted from real results. Lightweight is not claimed
to be negligible; use `level="none"` for an uninstrumented control and calibrate on
the target workload before using detailed timings for performance claims.

## Exported validation artifacts

Local generated artifacts are intentionally excluded from Git:

- `experiments/execution-validation/execution.html`: interactive measured preview.
- `experiments/execution-validation/execution-summary.pdf`: selected scope, one page.
- `experiments/execution-validation/{overview,epochs,kernels}.png`: browser captures.
- `experiments/execution-validation/full-report/analysis.html`: existing report plus link.
- `experiments/execution-validation/reopened/`: independently reopened export.
- `experiments/execution-overhead/overhead.json`: original local overhead experiment.

The explorer bounds embedded previews and rendered events. It is not a streaming
trace backend. Nsight, occupancy, SM/warp counters, bandwidth and exact per-node
hardware scheduling are unsupported. Full-mode CPU/GPU comparisons require
matching saved execution definitions; progressive cross-device speedup is withheld
until checkpoint-lineage matching has a separately validated protocol.

## Files and architecture

The common layer is `profiling/events.py` (journals/context), `profiling/accounting.py`
(exclusive timing) and `profiling/detail.py` (selected profiler sessions and CUDA
events). Native trainers and unchanged official factories emit through this layer.
`execution_analysis/` consumes saved results and traces; it never trains models.

Modified integration files:

- `config.py`, `benchmark/{runner,execution,matrix,execution_identity}.py`
- `datasets/{registry,validation}.py`, `models/registry.py` (observational wrappers)
- `profiling/{hardware,runtime,timing}.py`, `training/trainer.py`
- `progressive/{controller,graphs,reference,results,selection,training}.py`
- `runners/reference_worker.py`, `api/results.py`, `results/dashboard.py`
- `pyproject.toml` (packaged HTML asset), README and Python/profiling guides

New supporting files:

- `execution_analysis/{__init__,report,comparison}.py` and `viewer.html`
- `tests/test_execution_{events,profiles,report}.py`
- `examples/measure_execution_overhead.py`
- This guide, `execution_analysis.md`, `execution_analysis_design.md`,
  and `execution_overhead.json` under `docs/`

See the [usage guide](execution_analysis.md) for Full/Progressive commands,
disabling profiling, exported trace locations and offline reopening.
