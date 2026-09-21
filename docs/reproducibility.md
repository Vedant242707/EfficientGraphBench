# Reproducibility

Each measured result records package version, available Git commit, source hashes, model repository commit, installed software/environment specification, graph and split hashes, seed, model recipe, hardware fingerprint and profiler settings. Results from different comparison groups are never averaged by BenchmarkResults.summary(). Accuracy standard deviation across seeds uses sample standard deviation, consistent with CLI. Failed/skipped runs are retained and excluded from numerical averages. `--seeds 3` is seed ID 3.
