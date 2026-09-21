"""Inspectable registry metadata, independent of benchmark measurements."""

from efficientgraphbench.models.catalog import availability, model_spec
from efficientgraphbench.models.registry import MODEL_METADATA
from efficientgraphbench.names import MODEL_NAMES

PAPERS = {
    "gcn": "Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)",
    "graphsage": "Inductive Representation Learning on Large Graphs (NeurIPS 2017)",
    "gat": "Graph Attention Networks (ICLR 2018)",
    "gatv2": "How Attentive are Graph Attention Networks? (ICLR 2022)",
    "appnp": "Predict then Propagate: Graph Neural Networks meet Personalized PageRank (ICLR 2019)",
    "sgc": "Simplifying Graph Convolutional Networks (ICML 2019)",
    "graphgps": "Recipe for a General, Powerful, Scalable Graph Transformer (NeurIPS 2022)",
    "graphormer": "Do Transformers Really Perform Badly for Graph Representation? (NeurIPS 2021)",
    "sgformer": "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations (NeurIPS 2023)",  # noqa: E501
}


def get_model(name):
    name = name.lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    spec = model_spec(name)
    status, reason = availability(name)
    base = name.replace("_adapted", "").replace("_pyg", "").replace("_reference", "")
    return {
        **spec,
        **MODEL_METADATA[name],
        "model_id": name,
        "display_name": MODEL_NAMES[name],
        "implementation_source": spec.get("repository_url") or "EfficientGraphBench",
        "runner_type": spec["runner"],
        "supported_tasks": [] if name == "graphormer" else ["node_classification"],
        "paper": PAPERS.get(base),
        "status": status,
        "reason": reason,
    }


def list_models():
    return [get_model(name) for name in MODEL_NAMES]
