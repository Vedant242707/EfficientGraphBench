# EfficientGraphBench v0.1.0 — library implementation status

## Delivered

- Installable wheel and source distribution in dist/; src-layout and stable CLI entry point.
- Benchmark, ScalingBenchmark, BenchmarkResult, BenchmarkResults and HardwareProfile public APIs.
- Built-in and named-file custom dataset loading, descriptive InvalidDatasetError and optional numeric edge attributes.
- Shared Python/CLI matrix scheduling, failure results, deterministic pretraining skips and separate skip-after-OOM policy.
- Scaling returns only the final per-model result, retaining the last success after OOM; detailed size records remain saved.
- JSON, JSONL, CSV and Markdown exports; comparison groups kept separate.
- Packaged standalone reference worker, environment specs, unchanged hash-checked APPNP source and its upstream license.
- setup/environments/report/scaling commands; existing commands and scale alias retained.
- Model metadata inspection, version, provenance, documentation and complete PYTHON_COMMANDS.md.

## Verification

- 112 unit/regression tests passed; existing upstream PyTorch deprecation warning only.
- Ruff checks passed.
- Wheel and source distribution built successfully.
- Wheel installed into .package-test with separately installed core dependencies; dependency compatibility check passed (45 packages).
- Installed package exercised from a temporary directory outside the source checkout. GCN, APPNP, official SGFormer and official GraphGPS succeeded on a small graph; official Graphormer returned UNSUPPORTED_TASK. JSON/JSONL/CSV/Markdown exports passed.
- Existing GraphGPS and SGFormer installations passed new setup verification (clean pinned commits and model imports).
- Cora CUDA Python API smoke: GCN, SGFormer and GraphGPS, three seeds each, two epochs each: 9/9 SUCCESS. These are integration checks, not accuracy benchmarks.
- Editable development installation refreshed; existing egbench entry point resolves successfully.

Evidence: audits/library-release/wheel-smoke.json, cora-api-acceptance.json, model_provenance.json, build.log.

## Explicit boundaries

- Not published to PyPI. A public project license still needs to be selected before publication; no open-source licensing choice was made on the owner's behalf.
- Bundled research-environment installation locks are validated for Windows x86-64. Core package installation and CPU benchmark tests succeeded on this Windows machine; other platforms have not been validated.
- Setup verification reused the existing isolated reference environments; a complete second download/install of each research stack was not performed.
- Current task remains homogeneous transductive node classification. Graphormer reference remains UNSUPPORTED_TASK. No model architecture or profiler methodology was rewritten.
- --seeds 3 retains its original meaning (literal seed ID 3). Three repetitions use --seeds 1,2,3.
