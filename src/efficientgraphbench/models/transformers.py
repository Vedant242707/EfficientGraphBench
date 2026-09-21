"""Node-classification Transformer baselines using PyG operators."""

import torch
from torch_geometric.nn import GPSConv, SAGEConv


class GraphGPS(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        width = config.hidden_dim
        self.encoder = torch.nn.Linear(in_channels, width)
        self.blocks = torch.nn.ModuleList(
            [
                GPSConv(
                    width,
                    SAGEConv(width, width),
                    heads=config.heads,
                    dropout=config.dropout,
                    norm="layer_norm",
                    attn_type="performer",
                    attn_kwargs={"head_channels": width // config.heads, "dropout": config.dropout},
                )
                for _ in range(config.num_layers)
            ]
        )
        self.classifier = torch.nn.Linear(width, out_channels)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        # The benchmark supplies one full graph; every node belongs to that graph.
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for block in self.blocks:
            x = block(x, edge_index, batch=batch)
        return self.classifier(x)
