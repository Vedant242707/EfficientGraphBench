# Isolated environments

Reference Graph Transformers may require incompatible dependency stacks. EfficientGraphBench therefore runs them in isolated environments while controlling hardware, data, split, evaluation and profiling. The actual architectures are preserved.

Install Git and uv, then `egbench setup graphgps` or `egbench setup sgformer`. `egbench environments` shows readiness. Setup validates clean pinned source and imports; training is a separate benchmark. EGBENCH_HOME overrides the user cache. Existing development environments are recognized. Current bundled setup locks are validated for Windows x86-64 only. Graphormer setup reports UNSUPPORTED_TASK. No environment activation is needed.

Linux update: GraphGPS now has a Linux x86-64 installation path; see [GraphGPS on Linux](graphgps_linux.md). It requires verification on the target server. SGFormer automatic setup remains Windows-only.
