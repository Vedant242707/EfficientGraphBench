# Progressive refinement: implementation and validation

Full remains the default. There are exactly two protocols: Full and Progressive Multi-Fidelity. Progressive never launches a Full baseline automatically.

## File-by-file responsibilities

| File under `src/efficientgraphbench/` | Change |
|---|---|
| `results/protocol.py` | Reject mixed protocols in both directions; retain recognizable legacy records. |
| `api/results.py`, `results/writer.py` | Validate Full records before loading, aggregation and persistence. |
| `progressive/selection.py` | Require the exact expected seed set; retain deterministic validation-only ranking and uncertainty protection. |
| `progressive/controller.py` | Dispatch reference stages, preserve failed elapsed time, record selection counts and protocol policies, compare explicit baselines. |
| `profiling/timing.py` | Synchronize GPU timing boundaries and preserve phase duration on exceptions. |
| `profiling/runtime.py` | Attach measured resource peaks to failed attempts. |
| `training/trainer.py`, `benchmark/runner.py` | Separate Full optimization/validation timing, retain failure evidence, record execution identity and checkpoint policy. |
| `benchmark/matrix.py` | Save measured Full wall-clock envelope and exact run coverage. |
| `benchmark/execution_identity.py` | Fingerprint execution, environment, graph, recipe and hardware; exclude presentation-only sources. |
| `progressive/baseline.py` | Verify compatibility and reconcile optimization, evaluation, other execution and elapsed time without adding overlapping subsets. |
| `progressive/adaptation.py` | Distinguish first recovery from final target satisfaction and unnecessary recovery. |
| `progressive/training.py` | Continue model/optimizer/scheduler/RNG state; measure stage phases and partial failures. |
| `progressive/checkpoints.py` | Check lineage, hashes and restored model/optimizer signatures. |
| `progressive/graphs.py` | Preserve raw features in nested canonical graphs for reference recipes. |
| `progressive/reference.py` | Stage request/response bridge to pinned isolated reference environments. |
| `runners/progressive_worker.py` | Execute actual upstream SGFormer and GraphGPS models; regenerate GraphGPS structural preprocessing per stage. |
| `runners/reference_worker.py` | Dispatch Progressive stages while retaining existing Full execution. |
| `benchmark/interchange.py`, `benchmark/execution.py` | Verify canonical arrays, graph/split identity, reference results and checkpoint integrity. |
| `progressive/capabilities.py` | Explicitly enable tested official adapters; leave Graphormer unsupported. |
| `progressive/results.py`, `progressive/plots.py` | Separate finalists, eliminated candidates and failures; show recovered-only averages, transparent counts and measured charts. |
| `results/theme.py`, `results/dashboard.py`, `results/plots.py` | Shared dark HTML/portrait-print styling and dark Full charts. |

Additional files: `tests/test_progressive_refinement.py` supplies focused regressions and real installed-reference continuation tests; `configs/progressive_transformers.yaml` and `examples/progressive_transformers.py` provide reproducible GNN/transformer examples. `docs/progressive_benchmarking.md` and `PYTHON_COMMANDS.md` document usage and measurement policies.

## Model compatibility

| Model | Full | Progressive | Stage checkpoint resume |
|---|---|---|---|
| Official SGFormer | Supported with reference environment | Supported | Verified with actual pinned implementation on CPU |
| Official GraphGPS | Supported with reference environment | Supported | Verified with actual pinned implementation on CPU |
| SGFormer PyG | Supported, separate identity | Supported | Existing in-process continuation |
| Graphormer | Existing unsupported node-classification status preserved | Unsupported | Unsupported |

Reference tests expand a 20-node stage to 40 nodes and verify restored weight/optimizer signatures, parent checkpoint checksum, cumulative optimizer steps, implementation provenance and changed stage preprocessing identity. They do not substitute PyG SGFormer for the official model. Upstream source repositories are unchanged.

## Example

From the repository with its installed package and reference environments:

```shell
egbench progressive --config configs/progressive_transformers.yaml
```

Or run `python examples/progressive_transformers.py`. The configuration uses Texas, GCN, GraphSAGE, official SGFormer and official GraphGPS, three seeds, 50% then 100% graphs, and 10 then 20 additional epochs. `device: auto` selects available hardware; use `cuda` when GPU execution is required.

## Validation and interpretation

Final regression suite: **200 executed, 200 passed, 0 failed, 0 skipped**. Both installed official reference environments ran; no reference test was omitted for missing dependencies. One upstream TorchScript deprecation warning remains. Ruff checks passed after formatting cleanup.

The fresh report example used Texas with GCN, MLP, official SGFormer and official GraphGPS on CPU, one seed, and two plus two epochs. An explicitly executed four-epoch Full baseline supplied real elapsed metadata; no automatic Full run occurred. GraphGPS was eliminated in that particular example; its continuation was verified independently by the reference integration test.

Reports are in `output/pdf/EfficientGraphBench_Progressive_Dark.html` and `.pdf`, and `output/pdf/EfficientGraphBench_Full_Dark.html` and `.pdf`. PDFs were rendered and visually inspected for page overflow, backgrounds, chart colors and legibility. These tiny smoke measurements are not evidence of large-graph speedups or model quality.

Remaining limitations: official reference environments must be installed; CPU reference continuation is tested but CUDA continuation and large datasets need target-server validation. GraphGPS structural preprocessing and dense attention can still exhaust memory. Continuation operates between stages in one invocation; automatic restart after controller termination is not exposed. A forcibly killed worker preserves controller elapsed time but cannot return unfinished phase counters. Full and Progressive use different validation/checkpoint policies, so equal epoch budgets are not equal compute or graph exposure.
