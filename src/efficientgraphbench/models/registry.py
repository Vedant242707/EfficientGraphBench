import torch
from torch_geometric.nn import APPNP, SGConv
from torch_geometric.nn.models import GAT, GCN, GraphSAGE, SGFormer

from efficientgraphbench.config import ModelConfig
from efficientgraphbench.models.graphormer import Graphormer
from efficientgraphbench.models.reference_appnp import build_reference_appnp
from efficientgraphbench.models.transformers import GraphGPS
from efficientgraphbench.names import MODEL_NAMES

MODEL_METADATA = {
    "graphormer": {
        "family": "Graph Transformer",
        "attention": "global-dense-shortest-path-bias",
        "adaptation": "node adaptation; degree/spatial encodings; no graph token or typed edges",
    },
    "graphgps": {
        "family": "Graph Transformer",
        "attention": "local-plus-global-performer",
        "positional_encoding": "none",
        "local_operator": "GraphSAGE",
    },
    "mlp": {"family": "Feature baseline", "attention": "none"},
    "sgc": {"family": "GNN", "attention": "none"},
    "appnp": {"family": "GNN", "attention": "none"},
    "gatv2": {"family": "GNN", "attention": "local"},
    "gcn": {"family": "GNN", "attention": "none"},
    "graphsage": {"family": "GNN", "attention": "none"},
    "gat": {"family": "GNN", "attention": "local"},
    "sgformer": {"family": "Graph Transformer", "attention": "global-linear"},
}
MODEL_METADATA["graphgps_adapted"] = dict(MODEL_METADATA["graphgps"])
MODEL_METADATA["graphormer_adapted"] = dict(MODEL_METADATA["graphormer"])
MODEL_METADATA["sgformer_pyg"] = dict(MODEL_METADATA["sgformer"])
MODEL_METADATA["appnp_adapted"] = dict(MODEL_METADATA["appnp"])
MODEL_METADATA["sgformer_reference"] = dict(MODEL_METADATA["sgformer"])
MODEL_METADATA["graphgps"] = {"family": "Graph Transformer", "source": "official repository"}
MODEL_METADATA["graphormer"] = {"family": "Graph Transformer", "task": "unsupported"}
for identifier, metadata in MODEL_METADATA.items():
    metadata["name"] = MODEL_NAMES[identifier]


class FeatureMLP(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        layers = []
        dims = [in_channels] + [config.hidden_dim] * (config.num_layers - 1) + [out_channels]
        for index, (left, right) in enumerate(zip(dims, dims[1:])):
            layers.append(torch.nn.Linear(left, right))
            if index < config.num_layers - 1:
                layers.extend([torch.nn.ReLU(), torch.nn.Dropout(config.dropout)])
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index):
        return self.layers(x)


class APPNPModel(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.predict = FeatureMLP(config, in_channels, out_channels)
        self.propagate = APPNP(K=config.propagation_steps, alpha=config.alpha, cached=False)

    def forward(self, x, edge_index):
        return self.propagate(self.predict(x, edge_index), edge_index)


class GraphModel(torch.nn.Module):
    def __init__(self, model, data_interface=False):
        super().__init__()
        self.model = model
        self.data_interface = data_interface

    def forward(self, data):
        if self.data_interface:
            return self.model(data)
        if isinstance(self.model, Graphormer):
            return self.model(data)
        if isinstance(self.model, SGFormer):
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
            return self.model(data.x, data.edge_index, batch=batch)
        return self.model(data.x, data.edge_index)


def build_model(config: ModelConfig, in_channels: int, out_channels: int) -> GraphModel:
    common = dict(
        in_channels=in_channels, hidden_channels=config.hidden_dim, out_channels=out_channels
    )
    if config.name in {"graphormer", "graphgps", "sgformer", "sgformer_reference"}:
        raise ValueError("Reference models must be launched through the benchmark orchestrator")
    if config.name in {"graphormer_adapted", "graphgps_adapted"}:
        constructor = Graphormer if config.name == "graphormer_adapted" else GraphGPS
        model = constructor(config, in_channels, out_channels)
    elif config.name == "mlp":
        model = FeatureMLP(config, in_channels, out_channels)
    elif config.name == "sgc":
        # No cache: measured inference includes graph propagation on every call.
        model = SGConv(in_channels, out_channels, K=config.num_layers, cached=False)
    elif config.name == "appnp":
        return GraphModel(
            build_reference_appnp(config, in_channels, out_channels), data_interface=True
        )
    elif config.name == "appnp_adapted":
        model = APPNPModel(config, in_channels, out_channels)
    elif config.name == "sgformer_pyg":
        model = SGFormer(
            **common,
            trans_num_layers=1,
            trans_num_heads=1,
            trans_dropout=config.dropout,
            gnn_num_layers=config.num_layers,
            gnn_dropout=config.dropout,
        )
    else:
        constructors = {"gcn": GCN, "graphsage": GraphSAGE, "gat": GAT, "gatv2": GAT}
        if config.name not in constructors:
            raise ValueError(f"unsupported model: {config.name}")
        kwargs = {"heads": config.heads} if config.name in {"gat", "gatv2"} else {}
        if config.name == "gatv2":
            kwargs["v2"] = True
        model = constructors[config.name](
            **common, num_layers=config.num_layers, dropout=config.dropout, **kwargs
        )
    return GraphModel(model)
