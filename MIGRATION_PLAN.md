# Reference implementation migration

Prepared before implementation, 2026-09-20.

## Current audit

`benchmark/runner.py` directly loads every model into the controller's Python
process. GraphGPS is a local SAGE/Performer composition without positional
encodings. Graphormer is a local node classifier with a custom 4096-node guard.
SGFormer is PyG, not the authors' repository. GATv2 uses PyG GATv2Conv.
The comparison hash currently includes software and model hyperparameters;
this would incorrectly separate compatible reference implementations.
There is no Git repository in this checkout: a project commit cannot be invented.

## Retain

Retain dataset catalog, CSV/Excel/PDF preparation, validation, viewers, canonical
dataset loading, seed utilities, evaluators, existing model classes, and historical
experiments. Keep existing PyG model construction and training behavior available.
Never use old adapted results as evidence for reference models.

## Modify

Modify benchmark dispatch, config, model names/metadata, runtime/hardware profiling,
training budget metadata, result validation, reports, CLI, provenance generation,
and corresponding tests/docs. Preserve historical JSON aliases for report readers.
Do not overwrite historical experiments during integration validation.

## Phases and acceptance gates

1. Add BaseModelRunner, InProcessRunner, ExternalEnvironmentRunner, RepositoryRunner.
   Run subprocesses sequentially with argument arrays, independent logs and timeouts.
2. Versioned JSON requests/results; canonical NPZ numeric tensors and JSON index
   splits, SHA256 fingerprints. Validate identity, split, dataset and hardware on
   return; mismatches cannot become comparable successes.
3. Official GraphGPS checkout pinned to an observed commit, isolated Python3.10 /
   PyTorch1.13 / PyG2.2 environment as documented upstream. Adapter imports original
   GPSModel, encoders and preprocessing; config uses supported node head and PE/SE.
   Do not substitute local GraphGPS if dependencies fail. Record task configuration
   and recipe selection, which is not a tuned Cora paper reproduction.
4. Cora GraphGPS CUDA validation on actual detected GPU, three seeds, explicit
   preprocessing/train/inference/allocated/reserved measurements. Verify split and
   hardware consistency. Record OOM/dependency failures honestly.
5. Authors' SGFormer medium implementation, pinned checkout and separate environment.
   Preserve attention, graph branch and recommended recipe. Keep sgformer_pyg;
   expose sgformer_reference and label the selected primary implementation.
6. Cora and PubMed three-seed reference matrix, isolated experiment output directory.
7. Graphormer official task compatibility: upstream is molecular/graph prediction,
   with categorical atom/edge encoders and graph-token readout. Continuous-feature
   citation node classification requires more than a readout-only adapter. Primary
   Graphormer returns UNSUPPORTED_TASK pending a validated compatible reference;
   graphormer_adapted retains the custom 4096-node guard and implementation.
8. Scaling runner with explicit node/edge sizes, budgets and failure outcomes.
9. Recommendation filtering on measured primary successes within one hardware,
   dataset/split/protocol group; show budgets, no architecture-only claims.

## New files and interchange

Add benchmark execution/interchange modules, external runner entry points,
environments/{pyg,graphgps,sgformer,graphormer}/environment.yml and metadata.json,
reference repository checkouts, model-specific recipes, setup tooling, migration
tests. Environment metadata names its executable, repository and pinned commit.
Portable exchange uses NPZ x, edge_index, y, train_mask, val_mask, test_mask
(no pickle), with canonical JSON train/val/test node indices and SHA256 hashes.
Keep custom node mapping in the existing prepared dataset; shared indices refer
to exactly that canonical node order. Preserve available edge attributes.

## Result schema

Retain old fields; add model identity/category/source/repository commit, environment
specification, software versions, task/architecture changes, parameter counts,
dataset/split/hardware fingerprints, profiler settings, preprocessing time,
latency mean/median/std, allocated/reserved peaks, actual/max epochs, optimizer,
scheduler/patience/termination, source snapshot hashes and project commit or null.
Statuses: SUCCESS, OOM, UNSUPPORTED_TASK, UNSUPPORTED_GRAPH_SIZE, DEPENDENCY_ERROR,
CONFIG_ERROR, PREPROCESSING_ERROR, RUNTIME_ERROR. Failures retain graph sizes when
loading succeeded; unavailable values are null with reasons.
Strict groups depend on physical hardware, graph, split, task, metric and profiler
methodology, not framework versions or architecture-specific settings.

## Conflicts and risks

Official GraphGPS documents Python3.10, torch1.13 CUDA11.7, PyG2.2; controller
requires Python>=3.11 and newer torch. Old wheels, Windows support, CUDA architecture
support and GraphGym/Lightning APIs must be verified in isolation. Conda is absent;
uv/venv is available. Do not install old packages into the controller environment.
SGFormer needs compiled torch_sparse/torch_scatter dependencies. Graphormer uses
Fairseq and a distinct molecular pipeline. Inspect source before choosing pins.
Reference dense attention/PE preprocessing can OOM on PubMed; report rather than
silently changing architecture. Detect actual VRAM instead of assuming the supplied
8GB description. New model aliases, old result names, changed comparison hashes,
external process errors and config defaults are compatibility risks requiring tests.

## Primary sources consulted

- https://github.com/rampasek/GraphGPS (README environment and GPS recipe)
- https://github.com/qitianwu/SGFormer (authors' implementation)
- https://github.com/microsoft/Graphormer (molecular architecture/task pipeline)

Progress and validation results will be recorded separately; this plan is not a
claim that any integration or benchmark has already succeeded.
