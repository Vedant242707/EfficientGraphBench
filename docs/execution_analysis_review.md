# Execution Analysis review - 26 October 2026

This review distinguishes enabled settings, exported measurements and browser
features. No model architecture, optimization rule, checkpoint-selection policy
or progressive pruning rule was changed. Optional sampling and detailed profiling
have overhead and their timings are not uninstrumented performance claims.

## Supplied report diagnosis

The supplied `execution (1).html` has 101 sessions, all with `instrumentation: {}`,
zero traces and no capture-error notes. Empty options mean the default lightweight
mode. Its 18,000 epoch records and optimization/validation spans are consistent
with detailed profiling being disabled, not evidence of a failed CUDA exporter.
Its SHA-256 and the new measured evidence are in `execution_review_evidence.json`.
Existing exports cannot retrospectively recover operations or GPU kernels that
were never captured. The supplied file was not overwritten.

## Requirements and evidence

| Requirement | Implementation and verification |
|---|---|
| Detailed CUDA capture | GCN, Texas, CUDA, three epochs, seeds 1 and 2; epoch 2 profiled. Both attempts SUCCESS; each trace contains 382 CUDA kernels and 382 launch API calls. |
| Forward/loss/backward/update | Existing native and reference hooks activate in selected detailed epochs. Host wall intervals and current-stream CUDA event intervals are stored separately. Positive measurements verified for all four operations. No per-operation synchronization was added. |
| GPU inspection | Real kernel categories, launch calls, stream IDs, correlation IDs and timestamps exported by `torch.profiler`; browser kernel/activity filtering and event inspection passed. |
| Navigation | Model, seed, stage and epoch selectors; per-model attempt summary; optional controller/worker process inspection. Tested against the supplied 101-session export and new real traces. |
| Memory | Optional bounded background RSS/allocator queries during selected native epochs and reference attempts. Failure snapshots run before reference exception cleanup and after caught failure results. Journal recovery retains samples. Periodic samples, operation-boundary allocator values and legacy recorded peaks are separately labelled. |
| Portability | ZIP contains HTML, JSON and unabridged trace files with relative paths and SHA-256 hashes. Trace previews are limited to 2,000 intervals per trace and a shared 2 MiB serialized-event budget. Extracted-bundle trace download verified over localhost; unit test removes original source before reopening. |
| Failure preservation | Abrupt child-process exit, controlled worker MemoryError/OOM, native injected torch.OutOfMemoryError and interruption tests preserve completed phases and parent failed-attempt elapsed time. Incomplete child spans remain unknown rather than zero. |
| Progressive continuation | GCN/Texas, two stages, three additional epochs each. Both SUCCESS, each captured 237 kernels/launches. Second stage points to first checkpoint and records restoration: 0.0168826 s in this run. Stage selection verified in browser. |

Hardware tested: local NVIDIA GeForce RTX 4050 Laptop GPU; PyTorch 2.14.0+cu130,
CUDA build 13.0. These results do not establish profiling capability on the
college server until the smoke command is run there. Earlier development tests
had no CUDA kernels; this review supersedes that limitation for this tested
environment only, without asserting a retroactive cause for earlier failures.

Representative **actual** kernel names include `ampere_sgemm_128x32_nt`,
`ampere_sgemm_32x128_tn` and `ampere_sgemm_32x32_sliced1x4_nn`. The runtime trace
contains `cudaLaunchKernel`; one inspected launch/kernel correlation is 62 on
stream 7. Mangled kernel names are retained exactly, not replaced by invented
human-readable measurements.

## Timing semantics

Measured Full GCN/Texas seed 1, epoch 2:

| Operation | Host wall ms | Current-stream CUDA interval ms |
|---|---:|---:|
| Forward | 8.8177 | 8.8330 |
| Loss | 1.3659 | 1.3669 |
| Backward | 14.7413 | 3.9081 |
| Optimizer step | 1.9190 | 2.4436 |

Host intervals include Python/dispatch time. CUDA events are resolved at the end
of the selected epoch; they measure elapsed time on that stream, potentially
including idle gaps, not a sum of kernel execution times. They do not necessarily
cover work on other streams. These columns must not be added together. Profiler
kernel timestamps, CUDA event intervals and legacy synchronized training timers
have different boundaries. Profiling-start/export errors are recorded separately;
failed profiler startup does not skip the requested training work.

PyTorch documents that CUDA timing can exist without GPU kernel events in exported
JSON when CUPTI capture is unavailable: [PyTorch profiler documentation](https://docs.pytorch.org/docs/main/profiler).
Consequently, capability is established by inspecting exported events, not merely
by CUDA availability or the dashboard's existence.

## Exact API configuration

```python
from efficientgraphbench import Benchmark

results = Benchmark(
    dataset="texas", models=["gcn"], seeds=[1], epochs=3, device="cuda",
    output_dir="experiments/texas-detailed",
    profiling={
        "level": "detailed",
        "epochs": [2],
        "memory_sampling": {"interval_ms": 20, "max_samples": 50},
    },
).run()
results.execution_analysis()
```

For both Full and Progressive verification:

```bash
python examples/execution_analysis_smoke.py --device cuda
```

This script refuses to substitute CPU and exits with an explicit failure if no
CUDA kernel events were captured. `evidence.json` still records partial results.
It runs two Full seeds and two Progressive stages, not a large benchmark.

Epoch indices are one-based and restart at each progressive stage. Optional
`models`, `seeds` and `stages` lists restrict detailed profiling; stage names must
match configured fidelity stages. Omitting detailed mode intentionally retains
the lightweight behavior seen in the supplied report.

## Artifacts and checks

Generated local evidence (excluded from Git):

- `output/execution-review/gcn-texas-cuda-epoch2.trace.json`: full measured trace.
- `output/execution-review/full-execution-bundle.zip`: Full report and full traces.
- `output/execution-review/progressive-execution-bundle.zip`: progressive equivalent.
- `output/execution-review/{overview,epochs,kernels,progressive,supplied-navigation}.png`.
- `experiments/execution-review/texas/evidence.json`: complete smoke evidence.

Full regression suite: **235 passed**. Initial focused regression: 31 passed. Follow-up memory/reference/profile batch:
19 passed. Final report/diagnostic regression: 15 passed. Counts overlap. Includes CPU/CUDA official SGFormer and GraphGPS on
small generated validation inputs; this is not evidence of large-graph feasibility.
CPU checkpoint tensors remain identical across none/lightweight/detailed modes
in the controlled reproducibility test. Browser tests report zero JavaScript
errors; real launch/kernel details and relocated full-trace download were checked.

## Supported limits, not claims

- Memory sampling: interval 10-10,000 ms; cap 2-10,000 samples per process (default
  50 ms / 1,000). Tiny epochs can produce only start/end points. Samples miss
  transients and are not exact peak measurements. Once the cap is reached,
  subsequent history cannot be reconstructed; the limit is disclosed in notes.
- PyTorch allocator queries describe the current process, not all device users.
  They do not measure global device memory, SM occupancy, bandwidth or warp usage.
- Controller and child clocks remain separate. Their overlapping lifetimes are
  not summed. A forcibly killed child cannot produce an after-failure snapshot.
- HTML/JSON contain bounded previews. A lone HTML file does not contain all full
  traces. Transfer and extract the ZIP; in notebooks, download it using Jupyter's
  file browser. Kernel and runtime events are prioritized in previews, so a
  truncated preview is not a complete CPU/GPU schedule.
- No Nsight import, hardware counters, per-node/core assignment or GPU utilization
  claims. No physical GPU-memory exhaustion stress test was performed: failure
  tests deliberately simulate OOM without exhausting hardware.
