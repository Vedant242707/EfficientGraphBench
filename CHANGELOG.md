# Changelog

## Unreleased

- Download Cora, CiteSeer and PubMed directly from the upstream raw-content host to avoid GitHub response-header size errors, preserving PyG processing and splits.

- Add Progressive Multi-Fidelity with nested graph growth, checkpoint continuation, validation-only pruning, adaptation/recovery measurements and explicit Full-baseline compatibility.
- Add tested official SGFormer/GraphGPS reference-worker continuation; preserve SGFormer PyG as a separate identity and Graphormer's unsupported task status.
- Separate optimization/evaluation timing, validate protocol and seed coverage, and retain failed-attempt evidence.
- Add dataset profiling, classification metrics, Pareto comparisons and dark HTML/portrait-print reports with measured charts.
- Add college CUDA smoke and ten-seed PubMed Progressive configurations, plus Linux GraphGPS verification instructions.

- Remove custom Graphormer/GraphGPS model modules and adapted APPNP/MLP classes, registry entries, and obsolete tests. Maintained library/repository implementations and their interface wrappers remain.

- Replace the primary custom MLP baseline with `torch_geometric.nn.models.MLP` (ReLU, no normalization, linear output); retain truthful historical provenance for existing results. New MLP measurements require rerunning.

- Add OGB-Products and Roman Empire, Amazon Ratings, Actor, Texas, and Wisconsin loaders with recorded official split selection.
- Add `Benchmark(runs=10, splits="fixed" | "official" | "random")` and `egbench repeat`, split-aware summaries, and a college GPU Python/notebook guide.
- Record phase timing/memory snapshots, validation/checkpoint/test timing, and graph/parameter/gradient/optimizer tensor storage in both in-process and reference workers.
- Reuse checksummed graph exports across fixed-split runs and stream file checksums to avoid whole-file memory copies.

## 0.1.0

Public benchmark and scaling APIs, structured results and exports, packaged reference worker and APPNP source, isolated environment management, portable installed resources, and preserved node-classification CLI. Graphormer reference remains unsupported for the current task.
