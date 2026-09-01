#model.py
"""
Model definition for PertiNet protein-pair prediction and decision-margin scoring.
- Sequence branch: multi-scale 1D CNN → pooling → MLP.
- Structure branch: GVP-GNN over protein graphs, then pair pooling.
- Function branch: GO-term graph encoded by 2×GATv2; per-sample masked average.
- Fusion: separate modality tokens → Transformer interaction → sample-specific gate.
Outputs: logits (for binary prediction), disturbance score, and intermediate embeddings.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

try:  # package import (python -m ... from repository root)
    from .gvp import GVPConvLayer, LayerNorm
except ImportError:  # direct execution from model/
    from gvp import GVPConvLayer, LayerNorm


# ---------- helpers ----------
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
        self.residue_fc = nn.Sequential(
            nn.Linear(hidden * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.2),
        )

    def encode_residues(self, x: torch.Tensor) -> torch.Tensor:
        """Return one multiscale sequence embedding per residue."""
        x = x.transpose(1, 2)
        b3, b5, b7 = F.relu(self.c3(x)), F.relu(self.c5(x)), F.relu(self.c7(x))
        return self.residue_fc(torch.cat([b3, b5, b7], dim=1).transpose(1, 2))

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
    """Interaction-before-weighting fusion used in the manuscript.

    The three modality vectors remain separate tokens during Transformer
    interaction.  The complete model then estimates one softmax weight per
    modality and sample.  The other fusion types are matched controls used in
    the ablation study.
    """

    VALID_TYPES = {"token_transformer", "attention_only", "gated", "concat_mlp"}

    def __init__(
        self,
        d_seq: int,
        d_struct: int,
        d_func: int,
        out_dim: int = 128,
        fusion_type: str = "token_transformer",
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if fusion_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown fusion_type={fusion_type!r}")
        if out_dim % heads:
            raise ValueError("out_dim must be divisible by heads")
        self.fusion_type = fusion_type
        self.seq_proj = nn.Sequential(nn.Linear(d_seq, out_dim), nn.LayerNorm(out_dim))
        self.struct_proj = nn.Sequential(nn.Linear(d_struct, out_dim), nn.LayerNorm(out_dim))
        self.func_proj = nn.Sequential(nn.Linear(d_func, out_dim), nn.LayerNorm(out_dim))

        if fusion_type in {"token_transformer", "attention_only"}:
            self.modality_embedding = nn.Parameter(torch.empty(1, 3, out_dim))
            nn.init.normal_(self.modality_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=out_dim,
                nhead=heads,
                dim_feedforward=out_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=1)
        if fusion_type in {"token_transformer", "gated"}:
            self.evidence_gate = nn.Linear(out_dim, 1)
        if fusion_type == "concat_mlp":
            self.concat_mlp = nn.Sequential(
                nn.Linear(out_dim * 3, out_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 2, out_dim),
            )
        self.out_norm = nn.LayerNorm(out_dim)
        self.last_modality_weights = None

    def forward(self, h_seq, h_struct, h_func, modality_mask=None):
        h_seq, h_struct, h_func = map(_ensure_bd, (h_seq, h_struct, h_func))
        tokens = torch.stack(
            [self.seq_proj(h_seq), self.struct_proj(h_struct), self.func_proj(h_func)],
            dim=1,
        )
        if modality_mask is None:
            active = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            active = torch.as_tensor(modality_mask, dtype=torch.bool, device=tokens.device)
            if active.dim() == 1:
                active = active.unsqueeze(0).expand(tokens.size(0), -1)
            if active.shape != tokens.shape[:2] or not active.any(dim=1).all():
                raise ValueError("modality_mask must activate at least one of three modalities per sample")
        tokens = tokens * active.unsqueeze(-1)
        if self.fusion_type in {"token_transformer", "attention_only"}:
            tokens = self.token_encoder(
                tokens + self.modality_embedding * active.unsqueeze(-1),
                src_key_padding_mask=~active,
            )
            tokens = tokens * active.unsqueeze(-1)

        if self.fusion_type == "concat_mlp":
            self.last_modality_weights = None
            fused = self.concat_mlp(tokens.reshape(tokens.size(0), -1))
        elif self.fusion_type == "attention_only":
            weights = active.to(tokens.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
            self.last_modality_weights = weights.detach()
            fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        else:
            gate_logits = self.evidence_gate(tokens).squeeze(-1).masked_fill(~active, -torch.inf)
            weights = torch.softmax(gate_logits, dim=1)
            self.last_modality_weights = weights.detach()
            fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return self.out_norm(fused)


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
        self.use_seq = bool(cfg.get('use_sequence', True))
        self.use_struct = bool(cfg.get('use_structure', cfg.get('mode', 'full') == 'full'))
        self.use_go = bool(cfg.get('use_go', cfg.get('mode', 'full') != 'benchmark'))
        if self.use_struct:
            self.struct_enc = GVPEncoder(cfg['node_dims'], cfg['edge_dims'], 64)

        # fusion
        # struct branch yields 128-dim after pair pooling (i+j); if not used, we provide 128-dim dummy
        self.fusion = CrossModalFusion(
            64, 128, 64, out_dim=128,
            fusion_type=cfg.get('fusion_type', 'token_transformer'),
            heads=cfg.get('fusion_heads', 4),
            dropout=cfg.get('fusion_dropout', 0.2),
        )
        self.fusion_norm = nn.LayerNorm(128)

        # heads
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, cfg['num_labels'])
        )
        self.operating_offset = nn.Parameter(torch.tensor(0.5))

        # small dummies for ablations
        self.dummy_struct = nn.Parameter(torch.randn(1, 128) * 0.01)
        self.dummy_func   = nn.Parameter(torch.randn(1, 64) * 0.01)
        self.dummy_seq    = nn.Parameter(torch.randn(1, 64) * 0.01)

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
        if self.use_seq:
            h_seq = self.seq_enc(x['seq_feat'])              # [B, 64]
        else:
            h_seq = self.dummy_seq.expand(x['seq_feat'].size(0), -1)

        # structure
        if self.use_struct:
            h_nodes = self.struct_enc(x['x_s'], x['x_v'], x['edge_index'], x['edge_attr'])  # [N, 64]
            # pair pooling: take every two graphs per sample → concat
            # here batch indexes alternate 0,0,1,1,2,2,...
            # gather mean per component then concat
            # (assuming graphs were concatenated in that order upstream)
            # indices for pooling
            b = x['batch']
            # global maximum pooling by graph, matching the manuscript.
            num_graphs = int(b.max().item()) + 1
            pooled = torch.full(
                (num_graphs, h_nodes.size(-1)), -torch.inf,
                dtype=h_nodes.dtype, device=h_nodes.device,
            )
            pooled.scatter_reduce_(
                0, b.unsqueeze(-1).expand_as(h_nodes), h_nodes,
                reduce="amax", include_self=True,
            )
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            # pair-concat
            h_struct = torch.cat([pooled[0::2], pooled[1::2]], dim=-1)  # [B, 128]
        else:
            B = h_seq.size(0)
            h_struct = self.dummy_struct.expand(B, -1)                  # [B, 128]

        # function/GO
        if not self.use_go:
            h_func = self.dummy_func.expand(h_seq.size(0), -1)          # [B, 64]
        else:
            h_func = self.func_enc(x['go_feat'], x['go_edge_index'])    # [B, 64]

        # fusion & heads
        h = self.fusion(
            h_seq, h_struct, h_func,
            modality_mask=[self.use_seq, self.use_struct, self.use_go],
        )
        h = self.fusion_norm(h)
        logits = self.cls_head(h) - self.operating_offset               # [B, 1]
        prob   = torch.sigmoid(logits)
        decision_margin = (prob - 0.5).abs()                            # [B, 1]
        return logits, decision_margin, h, h_seq, h_struct, h_func, self.operating_offset


class PertiNetS(nn.Module):
    """Complementary residue/interface track using sequence and GVP features.

    `seq_feat` is [B, L, D] and `seq_mask` marks real residues. Structural node
    features must be provided in the same graph/residue order through the GVP
    fields used by :class:`PertiNet`.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.seq_enc = SequenceLocalEncoder(cfg["seq_input_dim"], 128, 64)
        self.struct_enc = GVPEncoder(cfg["node_dims"], cfg["edge_dims"], 64)
        self.interface_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )

    def forward(self, x: Dict):
        seq_res = self.seq_enc.encode_residues(x["seq_feat"])
        mask = x.get("seq_mask", torch.ones(seq_res.shape[:2], dtype=torch.bool, device=seq_res.device))
        seq_flat = seq_res[mask]
        struct_res = self.struct_enc(
            x["x_s"], x["x_v"], x["edge_index"], x["edge_attr"]
        )
        if seq_flat.size(0) != struct_res.size(0):
            raise ValueError("sequence-mask residues and structural graph nodes must align")
        return self.interface_head(torch.cat([seq_flat, struct_res], dim=-1)).squeeze(-1)
