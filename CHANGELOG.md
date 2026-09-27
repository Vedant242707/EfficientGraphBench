# Changelog

## Unreleased

- Remove custom Graphormer/GraphGPS model modules and adapted APPNP/MLP classes, registry entries, and obsolete tests. Maintained library/repository implementations and their interface wrappers remain.

- Replace the primary custom MLP baseline with `torch_geometric.nn.models.MLP` (ReLU, no normalization, linear output); retain truthful historical provenance for existing results. New MLP measurements require rerunning.

- Add OGB-Products and Roman Empire, Amazon Ratings, Actor, Texas, and Wisconsin loaders with recorded official split selection.
- Add `Benchmark(runs=10, splits="fixed" | "official" | "random")` and `egbench repeat`, split-aware summaries, and a college GPU Python/notebook guide.
- Record phase timing/memory snapshots, validation/checkpoint/test timing, and graph/parameter/gradient/optimizer tensor storage in both in-process and reference workers.
- Reuse checksummed graph exports across fixed-split runs and stream file checksums to avoid whole-file memory copies.

## 0.1.0

Public benchmark and scaling APIs, structured results and exports, packaged reference worker and APPNP source, isolated environment management, portable installed resources, and preserved node-classification CLI. Graphormer reference remains unsupported for the current task.
