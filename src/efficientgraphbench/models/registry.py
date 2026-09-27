import torch
from torch_geometric.nn import SGConv
from torch_geometric.nn.models import GAT, GCN, MLP, GraphSAGE, SGFormer

from efficientgraphbench.config import ModelConfig
from efficientgraphbench.models.reference_appnp import build_reference_appnp
from efficientgraphbench.names import MODEL_NAMES

MODEL_METADATA = {
    "graphormer": {"family": "Graph Transformer", "task": "unsupported"},
    "graphgps": {"family": "Graph Transformer", "source": "official repository"},
    "mlp": {"family": "Feature baseline", "attention": "none"},
    "sgc": {"family": "GNN", "attention": "none"},
    "appnp": {"family": "GNN", "attention": "none"},
    "gatv2": {"family": "GNN", "attention": "local"},
    "gcn": {"family": "GNN", "attention": "none"},
    "graphsage": {"family": "GNN", "attention": "none"},
    "gat": {"family": "GNN", "attention": "local"},
    "sgformer": {"family": "Graph Transformer", "attention": "global-linear"},
}
MODEL_METADATA["sgformer_pyg"] = dict(MODEL_METADATA["sgformer"])
MODEL_METADATA["sgformer_reference"] = dict(MODEL_METADATA["sgformer"])
for identifier, metadata in MODEL_METADATA.items():
    metadata["name"] = MODEL_NAMES[identifier]


class GraphModel(torch.nn.Module):
    def __init__(self, model, data_interface=False):
        super().__init__()
        self.model = model
        self.data_interface = data_interface

    def forward(self, data):
        if isinstance(self.model, MLP):
            return self.model(data.x)
        if self.data_interface:
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
    if config.name == "mlp":
        model = MLP(
            **common,
            num_layers=config.num_layers,
            dropout=float(config.dropout),
            act="relu",
            norm=None,
            plain_last=True,
        )
    elif config.name == "sgc":
        # No cache: measured inference includes graph propagation on every call.
        model = SGConv(in_channels, out_channels, K=config.num_layers, cached=False)
    elif config.name == "appnp":
        return GraphModel(
            build_reference_appnp(config, in_channels, out_channels), data_interface=True
        )
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
