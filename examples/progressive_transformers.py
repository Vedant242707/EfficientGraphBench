"""Small progressive example. Install official environments before running."""

from efficientgraphbench import ProgressiveBenchmark

if __name__ == "__main__":
    results = ProgressiveBenchmark(
        "texas",
        models=["gcn", "graphsage", "sgformer", "graphgps"],
        seeds=[1, 2, 3],
        device="auto",
        output_dir="experiments/progressive-transformers",
        progressive={
            "progression_strategy": "deterministic_nested",
            "selection_metric": "val_f1_macro",
            "min_candidates": 2,
            "stages": [
                {
                    "name": "initial",
                    "graph_fraction": 0.5,
                    "additional_epochs": 10,
                    "retain_fraction": 0.7,
                },
                {"name": "full", "graph_fraction": 1.0, "additional_epochs": 20},
            ],
        },
    ).run()
    results.show_all()
