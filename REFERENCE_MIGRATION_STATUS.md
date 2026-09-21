# Reference migration status

## Delivered

- Pre-implementation audit and migration plan: MIGRATION_PLAN.md.
- In-process, external-environment and pinned-repository runners.
- Versioned JSON exchange, portable numeric graphs, shared masks and fingerprints.
- Official GraphGPS and authors' SGFormer in isolated Python3.10/CUDA environments.
- Primary APPNP uses the unchanged pinned PyG citation Net class.
- Original prototype implementations remain under explicit adapted/library IDs.
- Unified failure records and preprocessing/latency/allocated/reserved profiling; hardware isolation.
- Environment/package locks, per-run source hashes and refreshed model provenance.
- Model listing, scaling sweeps, and measured-primary recommendation filtering.

## Validation

102 tests passed; the only warning is the installed PyTorch JIT deprecation warning.
Cora: 27 successful runs (nine models x three seeds); three Graphormer UNSUPPORTED_TASK results.
PubMed: 24 successful runs (eight models x three seeds), three GraphGPS timeout results, three Graphormer UNSUPPORTED_TASK results. Successful runs used 200 epochs.
All seeds within each dataset have matching graph/split/hardware/profiler fingerprints.
Scaling smoke: GCN, SGFormer and GraphGPS succeeded at 500 and 1000 nodes, seed 42, two epochs.

## Explicit limitations

GraphGPS has not completed the PubMed validation: its official dense LapPE preprocessing exceeded the declared 60-second worker budget. Results are RUNTIME_ERROR; no OOM or architectural node limit is inferred. Production timeout remains 7200 seconds, configurable in YAML.
The GraphGPS Actor node recipe is applied to citation data; it is not a tuned Cora/PubMed paper recipe.
SGFormer uses source architecture and optimizer settings with canonical public splits and explicit epoch budgets; this is not a claim of paper-score reproduction.
Graphormer has no validated unchanged reference adapter for this task. Its custom 4096 guard belongs only to graphormer_adapted.
PyG APPNP's original model class is unchanged; the benchmark uses its own canonical-split training/checkpoint protocol rather than the source script's 100-run loss-window harness.
The actual GPU reports 6141 MiB. NVML process VRAM was not measured; allocated/reserved metrics must not be presented as total process VRAM. The project Git commit is unavailable (null).

## Files

- experiments/reference-final/reports/cora.md
- experiments/reference-final/reports/pubmed.md
- experiments/reference-final/raw/results.jsonl and results.csv
- audits/review20/model_provenance.json
- audits/reference-migration/validation.json
- REFERENCE_BENCHMARKS.md and FUNCTION_REFERENCE.md
