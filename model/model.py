#model.py
"""
Model definition for PPI prediction + disturbance scoring.
- Sequence branch: multi-scale 1D CNN → pooling → MLP.
- Structure branch: GVP-GNN over protein graphs, then pair pooling.
- Function branch: GO-term graph encoded by 2×GATv2; per-sample masked average.
- Fusion: concat → tiny Transformer → MLP.
Outputs: logits (for binary prediction), disturbance score, and intermediate embeddings.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
# gvp-pytorch: https://github.com/drorlab/gvp-pytorch (assumed available)
from gvp import GVPConvLayer, LayerNorm


# ---------- helpers ----------
class SimpleTransformerBlock(nn.Module):
    """Single-token transformer (self-attn over a length-1 sequence)."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] → [B, 1, D]
        x1 = x.unsqueeze(1)
        out, _ = self.attn(x1, x1, x1)
        out = self.norm(out + x1)
        out = self.drop(out)
        return out.squeeze(1)  # [B, D]


def _ensure_bd(x: torch.Tensor) -> torch.Tensor:
    """Make sure tensor is [B, D]."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 3 and x.size(1) == 1:
        return x.squeeze(1)
    return x


# ---------- sequence encoder ----------
class SequenceLocalEncoder(nn.Module):
    """Three CNN branches (k=3/5/7) + (mean,max) pooling."""
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.c3 = nn.Conv1d(in_dim, hidden, 3, padding=1)
        self.c5 = nn.Conv1d(in_dim, hidden, 5, padding=2)
        self.c7 = nn.Conv1d(in_dim, hidden, 7, padding=3)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 6, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, in_dim]
        x = x.transpose(1, 2)  # [B, in_dim, L]
        b3, b5, b7 = F.relu(self.c3(x)), F.relu(self.c5(x)), F.relu(self.c7(x))
        def pool(b): return torch.cat([b.mean(-1), b.max(-1)[0]], dim=-1)
        x = torch.cat([pool(b3), pool(b5), pool(b7)], dim=-1)
        return self.fc(x)  # [B, out_dim]


# ---------- structure encoder (GVP) ----------
class GVPEncoder(nn.Module):
    """3×GVPConvLayer with LayerNorm. Returns per-node scalar embeddings ([N, s_dim])."""
    def __init__(self, node_dims: Tuple[int, int], edge_dims: Tuple[int, int], out_dim: int = 64):
        super().__init__()
        self.gvp1 = GVPConvLayer(node_dims, edge_dims, activations=(F.relu, None))
        self.n1 = LayerNorm(node_dims)
        self.gvp2 = GVPConvLayer(node_dims, edge_dims, activations=(F.relu, None))
        self.n2 = LayerNorm(node_dims)
        self.gvp3 = GVPConvLayer(node_dims, edge_dims, activations=(F.relu, None))
        self.n3 = LayerNorm(node_dims)
        self.proj = nn.Linear(node_dims[0], out_dim)

    def forward(self, x_s, x_v, edge_index, edge_attr):
        x_s, x_v = self.gvp1((x_s, x_v), edge_index, edge_attr)
        x_s, x_v = self.n1((x_s, x_v))
        x_s, x_v = self.gvp2((x_s, x_v), edge_index, edge_attr)
        x_s, x_v = self.n2((x_s, x_v))
        x_s, x_v = self.gvp3((x_s, x_v), edge_index, edge_attr)
        x_s, x_v = self.n3((x_s, x_v))
        return self.proj(x_s)  # [N, out_dim]


# ---------- GO function encoder ----------
class GOFunctionEncoder(nn.Module):
    """
    Encode a global GO graph with 2×GATv2, then aggregate per sample
    by multiplying sample multi-hot with the GO embeddings.
    """
    def __init__(self, go_input_dim: int, hidden: int = 64, out_dim: int = 64, num_go_terms: int = 2000):
        super().__init__()
        self.gat1 = GATv2Conv(go_input_dim, hidden, heads=2, concat=True)
        self.gat2 = GATv2Conv(hidden * 2, out_dim, heads=1)
        self.go_node_emb = nn.Parameter(torch.randn(num_go_terms, go_input_dim))

    def forward(self, go_multi_hot: torch.Tensor, go_edge_index: torch.Tensor) -> torch.Tensor:
        # Global GO encoding
        x = F.elu(self.gat1(self.go_node_emb, go_edge_index))
        x = F.elu(self.gat2(x, go_edge_index))              # [num_go, out_dim]
        # Sample-wise masked average
        weights = go_multi_hot.clamp(min=0.0)
        denom = weights.sum(dim=1, keepdim=True).add_(1e-6)
        return (weights @ x) / denom                         # [B, out_dim]


# ---------- fusion & heads ----------
class CrossModalFusion(nn.Module):
    def __init__(self, d_seq: int, d_struct: int, d_func: int, out_dim: int = 128):
        super().__init__()
        in_dim = d_seq + d_struct + d_func
        self.tiny_attn = SimpleTransformerBlock(in_dim, heads=4, dropout=0.2)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, 384), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(384, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, h_seq, h_struct, h_func):
        h_seq, h_struct, h_func = map(_ensure_bd, (h_seq, h_struct, h_func))
        h = torch.cat([h_seq, h_struct, h_func], dim=-1)
        h = self.tiny_attn(h)
        return self.fuse(h)  # [B, out_dim]


class DisturbanceRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):  # [B, in_dim] → [B, 1]
        return self.net(x)


# ---------- main model ----------
class PertiNet(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        # encoders
        self.seq_enc  = SequenceLocalEncoder(cfg['seq_input_dim'], 128, 64)
        self.func_enc = GOFunctionEncoder(cfg['go_input_dim'], 64, 64, cfg['num_go_terms'])
        self.use_struct = cfg.get('mode', 'full') == 'full'
        if self.use_struct:
            self.struct_enc = GVPEncoder(cfg['node_dims'], cfg['edge_dims'], 64)

        # fusion
        # struct branch yields 128-dim after pair pooling (i+j); if not used, we provide 128-dim dummy
        self.fusion = CrossModalFusion(64, 128, 64, out_dim=128)
        self.fusion_norm = nn.LayerNorm(128)

        # heads
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, cfg['num_labels'])
        )
        self.learnable_thr = nn.Parameter(torch.tensor(0.5))

        # small dummies for ablations
        self.dummy_struct = nn.Parameter(torch.randn(1, 128) * 0.01)
        self.dummy_func   = nn.Parameter(torch.randn(1, 64) * 0.01)

    def forward(self, x: Dict):
        """
        Expected x:
          - 'seq_feat': [B, L, seq_in]
          - 'go_feat' : [B, num_go_terms] (multi-hot)
          - 'go_edge_index': [2, E]
          - If structure enabled:
              'x_s','x_v','edge_index','edge_attr','batch' for both proteins flattened (torch_geometric Batch)
              Pair pooling rule: we assume batch alternates (p1_a, p1_b, p2_a, p2_b, ...)
        """
        # sequence
        h_seq = self.seq_enc(x['seq_feat'])                  # [B, 64]

        # structure
        if self.use_struct:
            h_nodes = self.struct_enc(x['x_s'], x['x_v'], x['edge_index'], x['edge_attr'])  # [N, 64]
            # pair pooling: take every two graphs per sample → concat
            # here batch indexes alternate 0,0,1,1,2,2,...
            # gather mean per component then concat
            # (assuming graphs were concatenated in that order upstream)
            # indices for pooling
            b = x['batch']
            # mean by graph id
            # torch_geometric doesn't ship a direct groupby mean here; do scatter_mean via segment
            num_graphs = int(b.max().item()) + 1
            pooled = torch.zeros(num_graphs, h_nodes.size(-1), device=h_nodes.device).index_add_(
                0, b, h_nodes
            )
            counts = torch.zeros(num_graphs, device=h_nodes.device).index_add_(0, b, torch.ones_like(b, dtype=torch.float))
            pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)  # [G, 64]
            # pair-concat
            h_struct = torch.cat([pooled[0::2], pooled[1::2]], dim=-1)  # [B, 128]
        else:
            B = h_seq.size(0)
            h_struct = self.dummy_struct.expand(B, -1)                  # [B, 128]

        # function/GO
        if x.get('mode', 'full') == 'benchmark':
            h_func = self.dummy_func.expand(h_seq.size(0), -1)          # [B, 64]
        else:
            h_func = self.func_enc(x['go_feat'], x['go_edge_index'])    # [B, 64]

        # fusion & heads
        h = self.fusion(h_seq, h_struct, h_func)
        h = self.fusion_norm(h)
        logits = self.cls_head(h) - self.learnable_thr                  # [B, 1]
        prob   = torch.sigmoid(logits)
        disturb = (prob - 0.5).abs()                                    # [B, 1]
        return logits, disturb, h, h_seq, h_struct, h_func, self.learnable_thr
