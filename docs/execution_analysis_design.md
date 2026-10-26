# Execution instrumentation: inspection and design

Work is isolated on `codex/execution-analysis`. New instrumentation lives in
`profiling/`; the optional explorer lives in `execution_analysis/`. Existing
reports and result artifacts are retained. No third benchmark mode is introduced.

## Phase 1: confirmed execution paths and gaps

Full: `Benchmark.run -> run_matrix -> run_benchmark -> _run_validated` loads and
exports the graph, then either calls native `train`/`measure_latency` or launches
`RepositoryRunner -> ExternalEnvironmentRunner -> reference_worker.execute`.
Progressive constructs nested graphs, calls native `train_stage` or
`train_reference_stage`, persists checkpoint lineage, then selects on validation.
The official progressive worker calls the same stage trainer with an upstream
factory. Reports aggregate saved records and do not train models.

Confirmed gaps before this change:

* The Full reference worker writes optimization/validation and preprocessing
  totals only after those phases complete. An exception before assignment loses
  partial measurements; its final JSON contains only the failure and available
  memory peaks. There is no incremental event file.
* Progressive's `train_stage` catches partial optimization/evaluation counters,
  but model initialization, LapPE preprocessing, conversion, transfers, imports
  and persistence are not independently measured. Its reference factory runs
  inside initialization before training.
* `train_reference_stage` assigns `worker_wall_time_sec` only after a successful
  return. The controller retains total failed elapsed, but not the internal
  breakdown of a timeout/crash. Full wraps worker lifetime in `finally`, but
  external-runner exceptions do not carry child events.
* Progressive accounting labels elapsed minus optimization/evaluation as
  "other execution". This is a remainder, not evidence that those operations
  were measured. Inclusive failed-attempt totals overlap other phases.
* `timed` synchronizes CUDA at existing optimization/evaluation boundaries.
  We must not add per-operation synchronization in ordinary runs.

GraphGPS evidence from the pinned checkout: `actor-GPS.yaml` uses two
`GCN+Transformer` layers, four heads and hidden size 64; LapPE is enabled with
four eigenvectors. The adapter calls upstream `compute_posenc_stats` on CPU
before model/data transfer. The Transformer backend is global dense attention.
These are code facts, not a measured attribution of the college OOM. Existing
college error text proves an allocation failure, but does not identify a stage
or allocation stack. New traces are required for those conclusions. No old
875/2456-second remainder can be retrospectively split into invented phases.

## Measurement contract

Events use process-local monotonic nanoseconds, a clock-domain identifier,
separate UTC timestamps, UUID identities and start/end append-only JSONL records.
Incomplete started spans remain incomplete after a crash, with unknown duration.
Parent subprocess lifetime is measured independently even without child output.
Different process clock domains are never silently aligned by wall timestamps.
Child breakdowns are shown separately, with startup/unobserved remainder;
inclusive parent and child durations are not summed.

Legacy scalar metrics retain their definitions. New execution schema version 1
contains exclusive phase accounting, partial failure evidence and explicit
unattributed time. Detailed runs carry a separate instrumentation identity and
must not mix with standard timings. Profiling never changes models, seeds,
splits, validation selection or epoch budgets.

GPU traces and CUDA-event intervals require actual local CUDA validation.
Hardware counters (occupancy/SM/warp/bandwidth) are unavailable unless an external
tool actually records them. Optional Nsight integration will be described as
unsupported where a reliable parser or tool is unavailable.
