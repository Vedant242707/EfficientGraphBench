"""Run from any directory after installing EfficientGraphBench."""
from efficientgraphbench import Benchmark

if __name__ == "__main__":
    results = Benchmark(dataset="cora", models=["gcn"], seeds=[42], epochs=2,
                        device="auto", output_dir="experiments/quickstart").run()
    print(results.summary())
    results.save_json("experiments/quickstart/results.json")
