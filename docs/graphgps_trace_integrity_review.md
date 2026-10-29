# GraphGPS trace integrity review

Reviewed 27 October 2026. Model architectures, training configuration, progressive pruning and pinned reference environments are unchanged.

## Confirmed export defect

The supplied Texas bundle declared 160 traces but contained 142 distinct trace files. GraphGPS contributed 20 references to two filenames. Eighteen declared SHA256 hashes did not match the packaged files.

The exporter flattened source paths to their basenames. Worker captures in separate attempt directories therefore overwrote one another during packaging. This establishes an export collision; it does not establish that the original worker files were overwritten. Earlier captures may remain in the college run's `exchange/*/execution-worker-epoch*.trace.json` files. They cannot be recovered from this ZIP alone.

## Fix and integrity checks

New capture filenames include benchmark, model, seed, stage, epoch, worker role, worker attempt identity and a unique capture ID. Files are reserved exclusively. Each capture embeds its identity, and saved metadata retains the same identity.

Export filenames also distinguish source directories and attempts, including legacy captures with identical basenames. The exporter checks identity and SHA256, rejects duplicate source references or conflicting destination files, and refuses to silently overwrite a capture. `trace-manifest.json` lists declared and packaged files. The completed ZIP is checked against that manifest. Reopening a legacy bundle with mismatched hashes reports an integrity error; regenerate it from intact original results and captures.

## Actual GraphGPS smoke test

Texas, official GraphGPS worker, CUDA, seeds 1 and 2, two epochs per seed, detailed profiling at both epochs:

- Both seeds completed successfully.
- Four references, four independently preserved captures and four packaged files; hashes and embedded seed identities verified.
- Worker Python 3.10.20, PyTorch 1.13.1+cu117, CUDA build 11.7, RTX 4050 Laptop GPU.
- Model parameters, node features and edges were on `cuda:0`.
- CPU and CUDA profiler activities were requested. This isolated Windows environment advertised only CPU activity support.
- Captures contained zero CUDA kernels/runtime events. CUDA operation-event intervals were measured, but these are not kernel traces.

The local test verifies preservation and CUDA execution, **not successful GraphGPS CUDA kernel capture**. No model replacement, CPU fallback or environment upgrade was applied.

The college archive advertised CPU and CUDA activities and contained RTX 4090 device properties, but no GraphGPS kernels or CUDA runtime events. The user subsequently supplied the corresponding worker log:

```text
function cbapi.getCuptiStatus() failed with error CUPTI_ERROR_NOT_INITIALIZED (15)
CUPTI initialization failed - CUDA profiler activities will be missing
```

This confirms CUPTI initialization failure as the reason for missing college CUDA activities, independently of the export collision. The log's subsequent permissions guidance is conditional: it does not report `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. The underlying reason CUPTI could not initialize remains unresolved and requires the college worker environment to investigate; neither CPU execution nor a permissions problem is established. New captures record executable/version, CUDA/Kineto capability, supported/requested activities, actual tensor devices and event counts. No college retest has yet demonstrated successful GraphGPS kernel capture.

## Run the verification on the college GPU

After updating the execution-analysis branch and activating its environment:

```bash
python examples/graphgps_trace_smoke.py --output experiments/graphgps-trace-audit-new
```

Equivalent benchmark configuration:

```python
from efficientgraphbench import Benchmark

results = Benchmark(
    dataset="texas", models=["graphgps"], seeds=[1, 2], epochs=2,
    device="cuda", output_dir="experiments/graphgps-trace-audit-new",
    profiling={"level": "detailed", "epochs": [1, 2]},
).run()
files = results.execution_analysis()
```

The script writes `trace-evidence.json` and `execution-analysis/execution-bundle.zip`. If kernel counts remain zero, inspect the diagnostics and `exchange/*/worker.log`. Success of training or HTML generation alone does not establish CUDA kernel profiling support.

## Native regression evidence

The native CUDA smoke test captured 382 real kernel events in each of two Full Benchmark seed captures, and 237 in each of two Progressive stage captures. Evidence is saved under `experiments/trace-identity-native-check/evidence.json`. These measurements are separate from the GraphGPS result.

Tests cover same-basename exports across seeds, duplicate paths, incorrect seed attribution, missing files, altered hashes, overwrite refusal, repeated epochs and legacy bundle corruption. These serialization fixtures do not count as measured GPU evidence.

Validation: full regression suite **243 passed** (one existing TorchScript deprecation warning); Ruff passed; Git whitespace checks passed. Final re-export of the four GraphGPS captures is under `experiments/graphgps-trace-audit/final-export/`.

## Follow-up: isolated CUPTI diagnostics (29 October 2026)

The launcher uses a fresh executable subprocess with no shell, redirects native stdout/stderr to its own worker log, and already removes newer PyTorch allocator settings unsupported by the pinned worker. It inherits library search paths. No CUDA libraries are injected by these changes. The launcher now exposes the log filename to the worker so selected-epoch profiling metadata can retain native CUPTI initialization errors. Generic conditional permission advice is excluded from actual error records.

Captures now include `gpu_profiling_status`, `cupti_errors`, scoped log evidence, and read-only library diagnostics. CPU traces survive absent CUDA events. The metadata embedded in the trace and the result both contain this information. Candidate libraries are distinguished from libraries actually mapped into the process; finding a CUPTI file does not prove it initialized.

Fresh-process local matrix-multiplication probe:

| Property | Main environment | GraphGPS worker |
| --- | --- | --- |
| Python | 3.11.15 | 3.10.20 |
| PyTorch | 2.14.0+cu130 | 1.13.1+cu117 |
| CUDA build | 13.0 | 11.7 |
| CUDA available | true | true |
| Kineto available | true | true |
| Advertised activities | CPU, CUDA | CPU |
| Loaded runtime DLL | cudart64_13.dll | cudart64_110.dll |
| Loaded CUPTI DLL | cupti64_2025.3.0.dll | cupti64_2022.2.0.dll |
| Measured CUDA kernels | 3 | 0 |

The main probe's measured kernel was `ampere_sgemm_128x32_nn`. Both runtime/CUPTI DLLs came from their respective environment's torch/lib directory. No cross-environment CUPTI loading was observed locally. CUDA build versions above are build identifiers, not runtime version queries; loaded filenames are additional evidence. Kineto's independent version is not exposed by the public PyTorch API, so diagnostics record that limitation plus the PyTorch build/git information instead of inventing a version.

The local worker's bundled CUPTI DLL exists and is loaded, but CUDA activity is not advertised and no kernels are exported. This does not reproduce or resolve the college Linux initialization error. NVIDIA documents incompatible driver/CUPTI combinations as one possible source of NOT_INITIALIZED; it is a hypothesis to check against actual college versions, not a diagnosed cause. See [NVIDIA CUPTI usage](https://docs.nvidia.com/cupti/main/main.html) and [PyTorch profiler fallback semantics](https://docs.pytorch.org/docs/stable/profiler.html).

On the college server, with these local changes copied/installed, run:

```bash
python examples/diagnose_graphgps_profiler.py
python examples/graphgps_trace_smoke.py --output experiments/graphgps-cupti-retest --require-cuda-kernels
```

The first command runs separate main/worker processes and saves `comparison.json`, stderr logs and uniquely named minimal traces under a fresh directory inside `experiments/graphgps-profiler-diagnostics/`. It records PATH, LD_LIBRARY_PATH, LD_PRELOAD, CUDA paths, installed candidates and mapped CUDA/CUPTI libraries. The second command verifies the real GraphGPS model and fails its final assertion if any requested capture lacks CUDA kernels, while retaining all available evidence and CPU traces.

College access requires the user's interactive SSH login. No college retest or successful GraphGPS CUDA kernel trace has been demonstrated by this follow-up. Do not change GraphGPS packages, provenance or architecture, replace CUPTI libraries, or grant profiling privileges based only on NOT_INITIALIZED.

Follow-up verification: 21 focused tests passed, including actual native/reference profiling, progressive profiling, trace identity/integrity and simulated CUPTI-log parsing. A fresh two-seed, two-epoch GraphGPS run again preserved all four traces and verified their ZIP checksums. Its embedded profiler metadata explicitly says `gpu_profiling_status: unavailable`; local `cupti_errors` is empty because this environment does not emit the college error. The supplied college error is tested as a log-parsing fixture, not claimed as a local observation. Evidence: `experiments/graphgps-cupti-review-oct29-verified/trace-evidence.json` and its `execution-analysis/execution-bundle.zip`.
