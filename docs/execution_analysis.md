# Execution Analysis

This feature lives on `codex/execution-analysis`. It complements Full and
Progressive benchmarks; it is not another benchmark mode. Existing prediction
metrics, model implementations, selection rules and checkpoint policies remain.

## Run a small test

```python
from efficientgraphbench import Benchmark

results = Benchmark(
    dataset="texas", models=["gcn"], seeds=[1], epochs=20, device="cuda",
    output_dir="experiments/execution-texas",
    profiling={"level": "detailed", "epochs": [1, 10]},
).run()
results.show_all()                    # Existing report, plus an execution link
results.execution_analysis()          # Optional standalone explorer
```

Use `device="cpu"` for a CPU-only test. Official models retain their separate,
pinned environments: install those using the existing setup commands first.
The same profiling option works with `sgformer` and `graphgps`; this does not
switch either model to a PyG substitute.

```python
from efficientgraphbench import ProgressiveBenchmark, ProgressiveConfig, FidelityStage

results = ProgressiveBenchmark(
    dataset="texas", models=["gcn", "graphgps"], seeds=[1], device="cuda",
    output_dir="experiments/execution-progressive",
    progressive=ProgressiveConfig(stages=[
        FidelityStage("initial", 0.5, 10),
        FidelityStage("full", 1.0, 10),
    ]),
    profiling={"level": "detailed", "models": ["graphgps"],
               "seeds": [1], "stages": ["full"], "epochs": [1, 5]},
).run()
results.show_all()
results.execution_analysis()
```

Epoch numbers are **one-based within each stage**, not cumulative checkpoint
epochs. Models use registry IDs; stage names must match the stage configuration.
For an explicit window, use `"epochs": list(range(10, 15))`. Each selected epoch
has a separate trace. Earlier training epochs provide ordinary workload warm-up;
this is not a claim that profiler startup overhead has been removed. Avoid
profiling every epoch of a large benchmark.

## Configuration

| Option | Meaning |
|---|---|
| Omitted, or `{"level": "lightweight"}` | Incremental lifecycle/epoch events; no operator trace or extra per-operation CUDA synchronization |
| `{"level": "none"}` | Disable new instrumentation; preserve existing benchmark timers/memory measurements |
| `{"level": "detailed", "epochs": [10]}` | Lifecycle events plus the selected epoch's operator trace and CUDA-event intervals when available |
| `models`, `seeds`, `stages` | Optional lists restricting detailed collection |

Detailed defaults to epoch 1 if `epochs` is omitted. Requesting a detailed run
marks its results instrumented even when selection excludes all epochs. Profiler
settings enter execution/comparison fingerprints, so those results cannot silently
join standard performance comparisons. Unknown options fail validation.

Lightweight instrumentation has measurable overhead, especially for tiny graphs;
see the validation report. Use `none` for an uninstrumented control experiment.
Do not interpret a profiled timing as the model's normal performance.

## Files and reopening

- `execution/*.jsonl`: append-only starts/ends; UUID IDs and process clock domains.
- `execution/*-lifecycle.summary.json`: controller lifecycle and exclusive accounting.
- `exchange/.../execution-worker.jsonl`: official worker journal, recoverable after a crash.
- `*-epochN.trace.json`: PyTorch Chrome traces, stored beside their event journal.
- `execution-analysis/execution.html`: offline dashboard.
- `execution-analysis/execution.json`: session, phase, epoch, failure, hardware and profiler metadata.
- `execution-analysis/traces/`: copied full traces accompanying the dashboard.

```python
from efficientgraphbench.execution_analysis import reopen_execution

report = reopen_execution("experiments/execution-texas/execution-analysis/execution.json")
print(report["report"])
```

In a Jupyter notebook, display the returned object (do not wrap its HTML in `HTML(...)`):

```python
from IPython.display import FileLink, display
display(FileLink(str(report["report"])))
```

The interactive report uses an isolated iframe so its scripts do not depend on
Jupyter's ordinary HTML output renderer. The links above it download the HTML and
JSON directly to your laptop; they do not depend on the notebook's folder. If
notebook trust settings block the iframe, download the HTML and open it in your
browser. Use its PDF button and choose **Save as PDF** in the print dialog.
The HTML/JSON contain trace previews; keep the original `traces/` directory for
full trace files. Reopening regenerates the page without rerunning the benchmark.

Reopening does not import or train a model. Keep the export directory together when
copying it to another computer. Open the HTML in a browser; it embeds a bounded
trace preview. The file picker can load the full saved Chrome trace. For an
external viewer, open Perfetto's trace viewer and select the exported trace JSON;
no benchmark rerun is needed. Actual supported fields depend on the exporter.

The **Save selected summary as PDF** button uses browser printing. It prints the
selected process's summary and phase/epoch tables rather than thousands of trace
events. Change the selected run or worker first. The HTML remains the detailed
interactive artifact.

## Reading the timings correctly

The original scalar timing definitions are unchanged. Execution schema 1 adds
separate, monotonic measurements. Each process has its own time origin, UTC labels
and event hierarchy. Exclusive wall accounting partitions covered intervals;
nested durations are not added. Concurrent siblings with ambiguous attribution
are shown as overlapping intervals. Incomplete spans have unknown duration, not
zero. Gaps remain explicitly unattributed.

The parent observes the worker attempt even when its process fails. Child spans
show imports, graph conversion, reference initialization, official LapPE, training
and evaluation where observable. **Parent and child totals are alternative scopes,
not additive categories.** Process startup before the child recorder begins and
shutdown after it ends cannot be assigned exact internal phases.

Per-operation wall measurements are CPU dispatch intervals. CUDA-event intervals
are measured on the current stream and resolved at selected-epoch completion;
they are neither CPU time nor sums of kernel durations. A selected epoch adds a
completion synchronization. Ordinary runs retain existing synchronization points.

Allocator observations are boundary samples, not continuous VRAM measurements.
The profiler may not know about tensors allocated before collection began. Peak
reserved memory is PyTorch allocator reservation, not total GPU usage. CPU RSS is
process memory, not incremental profiler overhead or CPU utilization.

## Verified boundaries and limitations

- Real CPU operator traces and real CUDA-event intervals were tested locally for
  GCN and both pinned official workers. The local Windows exporters produced
  **no CUDA kernel events**, despite successful CUDA training. The UI reports
  this explicitly. Linux/CUPTI kernel collection on the college server remains
  unverified; exported kernels are shown only if actually present.
- Occupancy, SM activity, warps, bandwidth, per-node CUDA-core assignment and
  Nsight import are unsupported. No estimate is presented as such a measurement.
- A trace lane uses actual process/thread/stream IDs. It does not establish
  hardware scheduling or GPU efficiency. CPU/GPU speedup requires matching saved
  workload provenance, budgets and profiler settings; unmatched runs produce none.
- Full-mode CPU/GPU comparisons are supported. Progressive CPU/GPU speedup is
  withheld because matching checkpoint lineage across devices needs an explicit
  protocol. Stage statistics remain available separately.
- Raw JSONL recovery tolerates a truncated final line. An abruptly killed child
  can lose its open span's duration and its unexported profiler buffer. The parent
  still retains attempted wall time. A machine power loss is not guaranteed
  durable: journal writes are flushed, not fsynced after every record.
- Failure events report the deepest observed failing phase and last completed
  epoch. They do not create a new recovery checkpoint. Existing progressive parent
  checkpoints remain governed by the existing checksum/continuation policy; an
  interrupted partial epoch is not resumable from these events alone.
- Detailed previews embed at most 20,000 intervals per trace and render at most
  300 matches. Full trace files retain the rest. Very large trace-file imports
  still require browser memory; this is not a streaming Perfetto replacement.
- Exact college GraphGPS OOM allocation stacks and the old unexplained remainders
  cannot be reconstructed from previous logs. Tiny validation graphs prove the
  instrumentation path, not large-graph feasibility. No architecture, dimension,
  graph fraction or attention backend is silently changed to avoid OOM.

See [design and confirmed timing gaps](execution_analysis_design.md) and
[validation evidence](execution_analysis_validation.md).
