# Dataset API

`from efficientgraphbench.datasets import load_dataset` returns DatasetBundle with `.data`, `.num_classes` and `.metadata()`. `.data` holds x, edge_index, y and train/val/test masks, plus optional attributes. `load_dataset("cora", root="data")` loads a built-in dataset. `egbench datasets` lists available IDs. A DatasetBundle can be passed to Benchmark or ScalingBenchmark. Model and split seeds are separate.
