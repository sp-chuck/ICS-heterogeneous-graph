from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for models/legacy_htgnn.py. "
        "Install it in your environment before using legacy_pyg backend."
    ) from exc


class SnapshotEncoder(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.dropout = dropout

        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_dim) for nt in self.node_types
        })

        conv_dict = {}
        for et in edge_types:
            conv_dict[et] = SAGEConv((-1, -1), hidden_dim)
        self.conv1 = HeteroConv(conv_dict, aggr="mean")

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {}
        for nt in self.node_types:
            x = self.input_proj[nt](data[nt].x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_dict[nt] = x

        # Some snapshots may miss a relation file; create empty edges for missing relation keys.
        for et in self.conv1.convs.keys():
            if et not in data.edge_types:
                src_t, _, _ = et
                empty = torch.empty((2, 0), dtype=torch.long, device=x_dict[src_t].device)
                data[et].edge_index = empty

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        pooled = []
        for nt in self.node_types:
            x = x_dict[nt]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            pooled.append(x.mean(dim=0))
        return torch.cat(pooled, dim=0)


class LegacyHTGNN(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int,
        gru_hidden: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = SnapshotEncoder(node_types, edge_types, hidden_dim, dropout)
        self.gru = nn.GRU(
            input_size=hidden_dim * len(node_types),
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.cls = nn.Linear(gru_hidden, 1)
        self.device_cls = nn.Linear(hidden_dim, 1)

    def forward(self, seq_data: List[HeteroData]) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = []
        last_device_emb = None
        for d in seq_data:
            graph_emb = self.encoder(d)
            embs.append(graph_emb)

        # Re-encode the last snapshot to get device-level embeddings for auxiliary supervision.
        last = seq_data[-1]
        x_dict = {}
        for nt in self.encoder.node_types:
            x = self.encoder.input_proj[nt](last[nt].x)
            x = F.relu(x)
            x = F.dropout(x, p=self.encoder.dropout, training=self.training)
            x_dict[nt] = x

        for et in self.encoder.conv1.convs.keys():
            if et not in last.edge_types:
                src_t, _, _ = et
                empty = torch.empty((2, 0), dtype=torch.long, device=x_dict[src_t].device)
                last[et].edge_index = empty

        x_dict = self.encoder.conv1(x_dict, last.edge_index_dict)
        if "device" in x_dict:
            last_device_emb = x_dict["device"]

        x = torch.stack(embs, dim=0).unsqueeze(0)
        out, _ = self.gru(x)
        snap_logit = self.cls(out[:, -1, :]).squeeze(-1)

        if last_device_emb is None:
            device_logit = torch.empty((0,), device=snap_logit.device)
        else:
            device_logit = self.device_cls(last_device_emb).squeeze(-1)
        return snap_logit, device_logit


__all__ = ["SnapshotEncoder", "LegacyHTGNN"]
