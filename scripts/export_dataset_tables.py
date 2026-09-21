"""Export locally cached benchmark tensors as plain CSV for viewing in Excel."""

import csv
from pathlib import Path

from efficientgraphbench.datasets.registry import load_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    for identifier, display in (("cora", "Cora"), ("citeseer", "CiteSeer"), ("pubmed", "PubMed")):
        cached = root / "data" / identifier / display / "processed" / "data.pt"
        if not cached.exists():
            raise FileNotFoundError(f"Dataset must already be downloaded: {cached}")
        data = load_dataset(identifier, root / "data").data
        output = root / "data" / "tabular" / display
        output.mkdir(parents=True, exist_ok=True)
        nodes = output / "nodes.csv"
        edges = output / "edges.csv"
        if nodes.exists() or edges.exists():
            raise FileExistsError(f"Export already exists: {output}")
        with nodes.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["node_id", "label", "split"]
                + [f"feature_{i + 1}" for i in range(data.num_node_features)]
            )
            features = data.x.numpy()
            labels = data.y.numpy()
            splits = ["unused"] * data.num_nodes
            for name in ("train", "val", "test"):
                for index in getattr(data, f"{name}_mask").nonzero().flatten().tolist():
                    splits[index] = name
            for index, row in enumerate(features):
                writer.writerow([index, int(labels[index]), splits[index], *row.tolist()])
        with edges.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source", "target"])
            writer.writerows(data.edge_index.t().tolist())
        # Check exported row counts and width against the graph used by the models.
        for path, count, width in (
            (nodes, data.num_nodes, data.num_node_features + 3),
            (edges, data.num_edges, 2),
        ):
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                assert len(next(reader)) == width
                actual = 0
                for row in reader:
                    assert len(row) == width
                    actual += 1
                assert actual == count
        print(f"{display}: {data.num_nodes} nodes, {data.num_edges} connections -> {output}")


if __name__ == "__main__":
    main()
