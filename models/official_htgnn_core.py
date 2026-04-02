from typing import Dict, List, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required for models/official_htgnn_core.py. "
        "Install it in your environment before using official_htgnn_core backend."
    ) from exc


class RelationAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(n_inp, n_hid),
            nn.Tanh(),
            nn.Linear(n_hid, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [num_nodes, num_relations, hidden_dim]
        w = self.project(h).mean(dim=0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((h.shape[0],) + beta.shape)
        return (beta * h).sum(dim=1)


class TemporalAgg(nn.Module):
    def __init__(self, n_inp: int, n_hid: int, max_len: int):
        super().__init__()
        self.proj = nn.Linear(n_inp, n_hid)
        self.q_w = nn.Linear(n_hid, n_hid, bias=False)
        self.k_w = nn.Linear(n_hid, n_hid, bias=False)
        self.v_w = nn.Linear(n_hid, n_hid, bias=False)
        self.fc = nn.Linear(n_hid, n_hid)
        pe = self._generate_positional_encoding(n_hid, max_len)
        self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _generate_positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros((max_len, d_model), dtype=torch.float)
        for i in range(max_len):
            for k in range(0, d_model, 2):
                div_term = math.exp(k * -math.log(100000.0) / d_model)
                pe[i][k] = math.sin((i + 1) * div_term)
                if k + 1 < d_model:
                    pe[i][k + 1] = math.cos((i + 1) * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim]
        h = self.proj(x)
        h = h + self.pe[: h.shape[1]].unsqueeze(0)
        q = self.q_w(h)
        k = self.k_w(h)
        v = self.v_w(h)
        score = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
        out = torch.matmul(score, v)
        return F.relu(self.fc(out))


class OfficialCoreSnapshotEncoder(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout

        self.input_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_dim) for nt in self.node_types
        })
        self.rel_aggr = RelationAgg(hidden_dim, hidden_dim)

        self.edge_key_to_type = {}
        self.rel_convs = nn.ModuleDict()
        for src_t, rel, dst_t in self.edge_types:
            key = f"{src_t}__{rel}__{dst_t}"
            self.edge_key_to_type[key] = (src_t, rel, dst_t)
            self.rel_convs[key] = SAGEConv((-1, -1), hidden_dim)

    def _prepare_edge_index(
        self,
        data: HeteroData,
        edge_type: Tuple[str, str, str],
        src_device: torch.device,
    ) -> torch.Tensor:
        if edge_type in data.edge_types:
            return data[edge_type].edge_index
        empty = torch.empty((2, 0), dtype=torch.long, device=src_device)
        data[edge_type].edge_index = empty
        return empty

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_in = {}
        for nt in self.node_types:
            x = self.input_proj[nt](data[nt].x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_in[nt] = x

        rel_msgs: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}
        for key, conv in self.rel_convs.items():
            src_t, _, dst_t = self.edge_key_to_type[key]
            edge_type = self.edge_key_to_type[key]
            edge_index = self._prepare_edge_index(data, edge_type, x_in[src_t].device)
            out = conv((x_in[src_t], x_in[dst_t]), edge_index)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            rel_msgs[dst_t].append(out)

        out_dict: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if not rel_msgs[nt]:
                out_dict[nt] = x_in[nt]
            elif len(rel_msgs[nt]) == 1:
                out_dict[nt] = rel_msgs[nt][0]
            else:
                stacked = torch.stack(rel_msgs[nt], dim=1)
                out_dict[nt] = self.rel_aggr(stacked)

        pooled = [out_dict[nt].mean(dim=0) for nt in self.node_types]
        return torch.cat(pooled, dim=0), out_dict


class OfficialCoreHTGNN(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[Tuple[str, str, str]],
        hidden_dim: int,
        gru_hidden: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.encoder = OfficialCoreSnapshotEncoder(node_types, edge_types, hidden_dim, dropout)
        self.temporal = TemporalAgg(
            n_inp=hidden_dim * len(node_types),
            n_hid=gru_hidden,
            max_len=max(seq_len, 1),
        )
        self.cls = nn.Linear(gru_hidden, 1)
        self.device_cls = nn.Linear(hidden_dim, 1)

    def forward(self, seq_data: List[HeteroData]) -> Tuple[torch.Tensor, torch.Tensor]:
        embs = []
        last_device_emb = None
        for i, d in enumerate(seq_data):
            graph_emb, node_emb_dict = self.encoder(d)
            embs.append(graph_emb)
            if i == len(seq_data) - 1:
                last_device_emb = node_emb_dict.get("device")

        seq_emb = torch.stack(embs, dim=0).unsqueeze(0)
        out = self.temporal(seq_emb)
        snap_logit = self.cls(out[:, -1, :]).squeeze(-1)

        if last_device_emb is None:
            device_logit = torch.empty((0,), device=snap_logit.device)
        else:
            device_logit = self.device_cls(last_device_emb).squeeze(-1)
        return snap_logit, device_logit


__all__ = [
    "RelationAgg",
    "TemporalAgg",
    "OfficialCoreSnapshotEncoder",
    "OfficialCoreHTGNN",
]
