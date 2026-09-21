"""Graphormer adaptation for continuous-feature, untyped node classification.

Uses degree embeddings and shortest-path attention biases. No graph token or
molecular edge-type encoder: the benchmark predicts nodes on untyped graphs.
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from efficientgraphbench.results.status import UnsupportedGraphSize

# Custom EfficientGraphBench safeguard, not an architectural Graphormer limit.
MAX_NODES = 4096
MAX_DISTANCE = 31
MAX_DEGREE = 511


def prepare_graphormer(data):
    if data.num_nodes > MAX_NODES:
        raise UnsupportedGraphSize(
            f"EfficientGraphBench safeguard: at most {MAX_NODES} nodes for this implementation; "
            f"received {data.num_nodes}. Dense attention requires quadratic memory.",
            {
                "owner": "EfficientGraphBench",
                "max_nodes": MAX_NODES,
                "enforced_by": "models/graphormer.py:prepare_graphormer",
            },
        )
    edges = data.edge_index.detach().cpu().numpy()
    graph = csr_matrix((np.ones(edges.shape[1]), edges), shape=(data.num_nodes, data.num_nodes))
    graph.sum_duplicates()
    graph.data[:] = 1
    distances = shortest_path(graph, directed=True, unweighted=True)
    spatial = np.where(
        np.isfinite(distances), np.minimum(distances, MAX_DISTANCE), MAX_DISTANCE + 1
    )
    device = data.x.device
    data.graphormer_spatial = torch.as_tensor(spatial, dtype=torch.long, device=device)
    data.graphormer_in_degree = torch.as_tensor(
        np.minimum(graph.getnnz(axis=0), MAX_DEGREE), dtype=torch.long, device=device
    )
    data.graphormer_out_degree = torch.as_tensor(
        np.minimum(graph.getnnz(axis=1), MAX_DEGREE), dtype=torch.long, device=device
    )
    return data


class GraphormerBlock(torch.nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(width)
        self.norm2 = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, width * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width * 4, width),
        )

    def forward(self, x, bias):
        normalized = self.norm1(x)
        attention, _ = self.attention(
            normalized, normalized, normalized, attn_mask=bias, need_weights=False
        )
        x = x + self.dropout(attention)
        return x + self.dropout(self.ffn(self.norm2(x)))


class Graphormer(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        width = config.hidden_dim
        self.encoder = torch.nn.Linear(in_channels, width)
        self.in_degree = torch.nn.Embedding(MAX_DEGREE + 1, width)
        self.out_degree = torch.nn.Embedding(MAX_DEGREE + 1, width)
        self.spatial = torch.nn.Embedding(MAX_DISTANCE + 2, config.heads)
        self.blocks = torch.nn.ModuleList(
            [GraphormerBlock(width, config.heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = torch.nn.LayerNorm(width)
        self.classifier = torch.nn.Linear(width, out_channels)

    def forward(self, data):
        if getattr(data, "graphormer_spatial", None) is None:
            prepare_graphormer(data)
        x = (
            self.encoder(data.x)
            + self.in_degree(data.graphormer_in_degree)
            + self.out_degree(data.graphormer_out_degree)
        ).unsqueeze(0)
        bias = self.spatial(data.graphormer_spatial).permute(2, 0, 1)
        for block in self.blocks:
            x = block(x, bias)
        return self.classifier(self.norm(x)).squeeze(0)
