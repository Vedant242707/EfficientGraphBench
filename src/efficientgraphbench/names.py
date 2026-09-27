"""Stable machine identifiers and correctly capitalized public names."""

from efficientgraphbench.datasets.catalog import DATASETS, canonical_dataset

MODEL_NAMES = {
    "graphormer": "Graphormer",
    "graphgps": "GraphGPS",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "sgformer": "SGFormer",
    "mlp": "MLP",
    "sgc": "SGC",
    "appnp": "APPNP",
    "gatv2": "GATv2",
    "sgformer_pyg": "SGFormer (PyG)",
    "sgformer_reference": "SGFormer (Reference)",
}
DATASET_NAMES = {key: spec.name for key, spec in DATASETS.items()}


def model_name(identifier: str) -> str:
    return MODEL_NAMES.get(identifier.lower(), identifier)


def dataset_name(identifier: str) -> str:
    return DATASET_NAMES.get(canonical_dataset(identifier), identifier)
