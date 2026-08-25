# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bmask(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x if x.dtype == torch.bool else x > 0.5


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = _bmask(mask).to(dtype=x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


@dataclass
class TRIAGEConfig:
    project_root: str = "data"
    pair_fourpack_dir: str = "data/TUnA/Intra1/fourpack"
    site_global_dir: str = "data/site_data/site_global"
    site_homo_dir: str = "data/site_data/HomoPDB_data"
    site_hetero_dir: str = "data/site_data/HeteroPDB_data"
    run_dir: str = "outputs"

    d_res_in: int = 1309
    d_chain_in: int = 0
    d_model: int = 256
    d_pair: int = 96
    n_encoder_layers: int = 2
    n_cross_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1

    use_film: bool = True
    use_structure_summary: bool = True
    use_l2_conv_refine: bool = True
    conv2d_channels: int = 16

    l1_kernel_size: int = 5
    topk: int = 32
    topm: int = 128
    block_size: int = 32
    block_topr: int = 4
    l2_gate_alpha: float = 0.75
    l2_geom_tau: float = 8.0
    l2_geom_beta_init: float = 0.25

    lr: float = 2e-4
    l1_lr: float = 1e-4
    l1_warmup_epochs: int = 2
    l1_scheduler_epochs: int = 80
    l1_min_lr_factor: float = 0.30
    l1_ema_decay: float = 0.0
    weight_decay: float = 1e-4
    batch_size: int = 2
    l1_batch_size: int = 8
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    min_val_items: int = 1
    num_workers: int = 0
    l1_num_workers: int = 4
    max_pair_len: int = 512
    max_site_len: int = 768
    epochs_debug: int = 1
    epochs_struct_pretrain: int = 50
    epochs_pair_fusion: int = 30
    epochs_joint_finetune: int = 30
    epochs_l1_graphrbf: int = 80
    patience_debug: int = 0
    patience_struct_pretrain: int = 0
    patience_pair_fusion: int = 0
    patience_joint_finetune: int = 0
    patience_l1_graphrbf: int = 12
    pp_root: str = "data/Dset_prepared"
    pp_esm_dir: str = "data/esm_embeddings/graphrbf_pp"
    dest_raw_root: str = "data/Dset"
    dest_root: str = "data/Dset_prepared"
    dest_esm_dir: str = "data/esm_embeddings/dset"
    dest_base_checkpoint: str = "checkpoints/triage_final.pt"
    dest_pairing_mode: str = "same_pdb_or_self"
    dest_pair_max_partners: int = 1
    rbp400_root: str = "data/RBP400"
    rbp400_id_list: str = "data/RBP400/accessions.txt"
    rbp400_esm_dir: str = "data/esm_embeddings/rbp400"
    joint_struct_steps: int = 2
    joint_pair_steps: int = 1
    grad_clip: float = 1.0

    w_res: float = 1.0
    l1_pos_weight_cap: float = 10.0
    l1_label_smoothing: float = 0.0
    w_l1_rank: float = 0.10
    l1_rank_margin: float = 0.20
    l1_rank_start_epoch: int = 1
    l1_rank_ramp_epochs: int = 1
    l1_rank_max_pairs: int = 2048
    w_l1_hard_rank: float = 0.0
    l1_hard_rank_margin: float = 0.50
    l1_hard_rank_neg_frac: float = 0.20
    l1_hard_rank_max_neg: int = 128
    l1_hard_rank_start_epoch: int = 1
    l1_hard_rank_ramp_epochs: int = 1
    w_l1_topband_bce: float = 0.0
    l1_topband_frac: float = 0.20
    l1_topband_min_k: int = 10
    l1_topband_max_k: int = 128
    l1_topband_start_epoch: int = 1
    l1_topband_ramp_epochs: int = 1
    w_l1_l10_boundary: float = 0.0
    l1_l10_boundary_frac: float = 0.10
    l1_l10_boundary_margin: float = 0.04
    l1_l10_boundary_max_pos: int = 64
    l1_l10_boundary_start_epoch: int = 1
    l1_l10_boundary_ramp_epochs: int = 1
    l1_score_w_auc: float = 0.70
    l1_score_w_auprc: float = 0.10
    l1_score_w_mcc: float = 0.15
    l1_score_w_f1: float = 0.05
    l1_score_w_acc: float = 0.0
    l1_score_loss_penalty: float = 0.0
    l1_score_w_recall_l5: float = 0.45
    l1_score_w_recall_l10: float = 0.35
    l1_score_w_precision_10: float = 0.10
    l1_score_w_hit_2: float = 0.10
    l1_score_w_hit_20: float = 0.0
    rbp400_triage_score_w_pair_auprc: float = 0.05
    rbp400_triage_score_w_pair_mcc: float = 0.03
    rbp400_triage_score_w_recall_l5: float = 0.30
    rbp400_triage_score_w_recall_l10: float = 0.30
    rbp400_triage_score_w_precision_10: float = 0.25
    rbp400_triage_score_w_hit_20: float = 0.05
    rbp400_triage_score_w_gate_entropy: float = 0.02
    rbp400_triage_score_loss_penalty: float = 0.01
    l1_per_protein_loss: bool = False
    l1_extreme_label_weight: float = 1.0
    l1_zero_label_weight: float = -1.0
    l1_full_label_weight: float = -1.0
    l1_exclude_zero_label_proteins: bool = False
    l1_ager_enable: bool = False
    l1_ager_radius: float = 10.0
    l1_ager_alpha: float = 0.30
    l1_ager_top_m: int = 5
    l1_single_chain_mode: bool = True
    use_l1_raw_skip: bool = True
    l1_raw_skip_alpha: float = 0.50
    use_l1_multiscale_head: bool = False
    l1_multiscale_channels: int = 64
    l1_multiscale_delta_init: float = 0.05
    use_l1_geom_adapter: bool = False
    l1_geom_alpha: float = 0.15
    use_l1_geom_early: bool = False
    l1_geom_early_alpha: float = 0.25
    l1_threshold_mode: str = "auto_mcc"
    l1_threshold_min_recall: float = 0.0
    l1_threshold_min_precision: float = 0.0
    l1_threshold_beta: float = 1.50
    dest_balanced_res_loss: bool = False
    w_contact: float = 1.0
    w_l1_l2_consistency: float = 0.25
    w_topk_contact_ranking: float = 0.25
    w_topk_margin_rank: float = 0.20
    topk_rank_margin: float = 0.50
    w_triage_struct: float = 0.75
    w_triage_pair: float = 1.0
    w_l2_mil_weak: float = 0.25
    w_interface_pair_consistency: float = 0.20
    w_negative_evidence_suppression: float = 0.10
    w_gate_regularization: float = 0.02
    w_reliability: float = 0.02
    w_struct_gate_regularization: float = 0.25
    struct_max_pair_gate: float = 0.45
    struct_min_local_gate: float = 0.35
    max_pair_gate: float = 0.85
    min_gate_entropy: float = 0.80
    use_pair_pssm: bool = True
    use_pair_dssp_ss: bool = True
    use_pair_dssp_rsa: bool = False
    use_site_pssm: bool = True
    use_site_dssp_ss: bool = True
    use_site_dssp_rsa: bool = True
    use_esm: bool = True
    pair_esm_dir: str = "data/esm_embeddings/pair"
    site_esm_dir: str = "data/esm_embeddings/site"
    use_coords_geometry: bool = True
    pair_rsa_available_rate: float = 0.0145513338722716
    site_pssm_available_rate: float = 0.9239130434782609
    site_rsa_available_rate: float = 1.0

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> Dict:
        return asdict(self)


class FiLM(nn.Module):
    def __init__(self, d_model: int, d_chain: int):
        super().__init__()
        self.enabled = d_chain > 0
        dc = max(1, d_chain)
        self.net = nn.Sequential(
            nn.Linear(dc, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x: torch.Tensor, chain: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.enabled or chain is None:
            return x
        if chain.dim() == 3:
            chain = chain.mean(dim=1)
        gamma, beta = self.net(chain).chunk(2, dim=-1)
        return x * (1.0 + 0.5 * torch.tanh(gamma).unsqueeze(1)) + 0.1 * beta.unsqueeze(1)


class SelfEnc(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        if layers <= 0:
            self.mod = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=d * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mod = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.mod, nn.Identity):
            return x
        kpm = None if mask is None else ~_bmask(mask)
        return self.mod(x, src_key_padding_mask=kpm)


class CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ca = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cb = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.nqA = nn.LayerNorm(d)
        self.nkA = nn.LayerNorm(d)
        self.nqB = nn.LayerNorm(d)
        self.nkB = nn.LayerNorm(d)
        self.ffa = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.ffb = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, xA, xB, maskA=None, maskB=None):
        kpmA = None if maskA is None else ~_bmask(maskA)
        kpmB = None if maskB is None else ~_bmask(maskB)
        hA, _ = self.ca(self.nqA(xA), self.nkB(xB), self.nkB(xB), key_padding_mask=kpmB, need_weights=False)
        hB, _ = self.cb(self.nqB(xB), self.nkA(xA), self.nkA(xA), key_padding_mask=kpmA, need_weights=False)
        xA = xA + self.drop(hA)
        xB = xB + self.drop(hB)
        return xA + self.drop(self.ffa(xA)), xB + self.drop(self.ffb(xB))


class CrossEnc(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([CrossBlock(d, heads, dropout) for _ in range(layers)])

    def forward(self, xA, xB, maskA=None, maskB=None):
        for blk in self.layers:
            xA, xB = blk(xA, xB, maskA, maskB)
        return xA, xB


class ResidueEvidenceHead(nn.Module):
    def __init__(self, d: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel_size, padding=pad, groups=1)
        self.mlp = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        z = self.norm(x).transpose(1, 2)
        z = F.gelu(self.conv(z)).transpose(1, 2)
        logits = self.mlp(z).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~_bmask(mask), -30.0)
        return logits


class MultiScaleResidueEvidenceHead(nn.Module):
    """L1 residue head with adaptive 3/7/15-residue local evidence."""

    def __init__(self, d: int, channels: int, dropout: float, delta_init: float = 0.05):
        super().__init__()
        c = max(16, int(channels))
        self.base = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, c)
        self.convs = nn.ModuleList([nn.Conv1d(c, c, k, padding=k // 2) for k in (3, 7, 15)])
        self.scale_gate = nn.Linear(d, 3)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c, 1),
        )
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_init)))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        base = self.base(x)
        h = self.in_proj(self.norm(x)).transpose(1, 2)
        scales = torch.stack([conv(h).transpose(1, 2) for conv in self.convs], dim=2)
        weights = torch.softmax(self.scale_gate(x), dim=-1).unsqueeze(-1)
        local = (scales * weights).sum(dim=2)
        logits = (base + torch.tanh(self.delta_scale) * self.out_proj(local)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~_bmask(mask), -30.0)
        return logits


class LiteConv2DRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s + self.net(s.unsqueeze(1)).squeeze(1)


class StructureSummary(nn.Module):
    """Small GVP-inspired geometry summary for L3 context when C-alpha coords exist."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, coords: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if coords is None:
            return None
        if not torch.is_tensor(coords):
            return None
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.dim() != 3 or coords.size(-1) < 3:
            return None
        coords = coords[..., :3].contiguous()
        if mask is not None:
            if not torch.is_tensor(mask):
                return None
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() != 2:
                return None
            mask = mask.to(device=coords.device)
            if coords.size(0) != mask.size(0) or coords.size(1) != mask.size(1):
                return None
        valid = torch.isfinite(coords).all(dim=-1)
        if mask is not None:
            valid = valid & _bmask(mask)
        valid = valid.contiguous()
        if valid.dim() != 2 or valid.size(0) != coords.size(0) or valid.size(1) != coords.size(1):
            return None
        if not bool(valid.any().detach().cpu().item()):
            return None
        c = torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)
        denom = valid.sum(1, keepdim=True).clamp_min(1.0).to(dtype=c.dtype)
        center = (c * valid.unsqueeze(-1).to(dtype=c.dtype)).sum(1) / denom
        centered = c - center.unsqueeze(1)
        radius = torch.sqrt((centered.square().sum(-1) * valid.to(dtype=c.dtype)).sum(1, keepdim=True) / denom)
        B, L = valid.shape
        pos = torch.arange(L, device=valid.device, dtype=torch.long)
        valid_long = valid.to(dtype=torch.long)
        first_idx = valid_long.argmax(dim=1)
        last_idx = (valid_long * pos.unsqueeze(0)).argmax(dim=1)
        batch_idx = torch.arange(B, device=c.device)
        first = c[batch_idx, first_idx]
        last = c[batch_idx, last_idx]
        has_valid = valid.any(dim=1, keepdim=True)
        first = torch.where(has_valid, first, torch.zeros_like(first))
        last = torch.where(has_valid, last, torch.zeros_like(last))
        direction = F.normalize(last - first, dim=-1, eps=1e-6)
        feat = torch.cat([radius, center.norm(dim=-1, keepdim=True), direction, valid.float().mean(1, keepdim=True)], dim=-1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return self.proj(feat)


def select_topk_residues(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    k: int,
    block_size: Optional[int] = None,
    block_topr: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # This implementation keeps the block-wise API but uses robust global Top-K.
    del block_size, block_topr
    masked = logits if mask is None else logits.masked_fill(~_bmask(mask), -1e9)
    kk = max(1, min(int(k), masked.size(1)))
    return torch.topk(masked, kk, dim=1, largest=True, sorted=True)


def select_topm_interface(
    scores: torch.Tensor,
    maskA: Optional[torch.Tensor],
    maskB: Optional[torch.Tensor],
    m: int,
    block_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # This implementation keeps the block-wise API but uses global sparse Top-M support.
    del block_size
    B, LA, LB = scores.shape
    valid = torch.ones((B, LA, LB), dtype=torch.bool, device=scores.device)
    if maskA is not None:
        valid = valid & _bmask(maskA).unsqueeze(-1)
    if maskB is not None:
        valid = valid & _bmask(maskB).unsqueeze(1)
    flat = scores.masked_fill(~valid, -1e9).reshape(B, LA * LB)
    mm = max(1, min(int(m), flat.size(1)))
    vals, flat_idx = torch.topk(flat, mm, dim=1, largest=True, sorted=True)
    idx = torch.stack([torch.div(flat_idx, LB, rounding_mode="floor"), flat_idx % LB], dim=-1)
    return idx, vals


def project_interface_to_residue(
    S_interface: torch.Tensor,
    maskA: Optional[torch.Tensor],
    maskB: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = torch.sigmoid(S_interface)
    if maskB is not None:
        pA = p.masked_fill(~_bmask(maskB).unsqueeze(1), -1.0).amax(dim=2).clamp_min(0.0)
    else:
        pA = p.amax(dim=2)
    if maskA is not None:
        pB = p.masked_fill(~_bmask(maskA).unsqueeze(2), -1.0).amax(dim=1).clamp_min(0.0)
    else:
        pB = p.amax(dim=1)
    return pA, pB


class TRIAGEPPIModel(nn.Module):
    def __init__(self, cfg: TRIAGEConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.projA = nn.Sequential(nn.Linear(cfg.d_res_in, D), nn.GELU(), nn.LayerNorm(D))
        self.projB = nn.Sequential(nn.Linear(cfg.d_res_in, D), nn.GELU(), nn.LayerNorm(D))
        self.filmA = FiLM(D, cfg.d_chain_in if cfg.use_film else 0)
        self.filmB = FiLM(D, cfg.d_chain_in if cfg.use_film else 0)
        self.l1_geom_embed_A = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )
        self.encA = SelfEnc(D, cfg.n_encoder_layers, cfg.n_heads, cfg.dropout)
        self.encB = SelfEnc(D, cfg.n_encoder_layers, cfg.n_heads, cfg.dropout)
        self.cross = CrossEnc(D, cfg.n_cross_layers, cfg.n_heads, cfg.dropout)

        head_cls = MultiScaleResidueEvidenceHead if bool(getattr(cfg, "use_l1_multiscale_head", False)) else ResidueEvidenceHead
        if head_cls is MultiScaleResidueEvidenceHead:
            self.res_head_A = head_cls(D, getattr(cfg, "l1_multiscale_channels", 64), cfg.dropout, getattr(cfg, "l1_multiscale_delta_init", 0.05))
            self.res_head_B = head_cls(D, getattr(cfg, "l1_multiscale_channels", 64), cfg.dropout, getattr(cfg, "l1_multiscale_delta_init", 0.05))
        else:
            self.res_head_A = head_cls(D, cfg.l1_kernel_size, cfg.dropout)
            self.res_head_B = head_cls(D, cfg.l1_kernel_size, cfg.dropout)
        self.l1_raw_head_A = nn.Sequential(
            nn.LayerNorm(cfg.d_res_in),
            nn.Linear(cfg.d_res_in, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, 1),
        )
        self.l1_geom_adapter_A = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, max(16, D // 2)),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(max(16, D // 2), 1),
        )
        self.res_vec = nn.Sequential(nn.LayerNorm(D * 2 + 6), nn.Linear(D * 2 + 6, D), nn.GELU(), nn.Dropout(cfg.dropout))
        self.res_logit = nn.Linear(D, 1)

        self.l2_projA = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, cfg.d_pair))
        self.l2_projB = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, cfg.d_pair))
        self.l2_geom_beta = nn.Parameter(torch.tensor(float(cfg.l2_geom_beta_init)))
        self.l2_refine = LiteConv2DRefine(cfg.conv2d_channels) if cfg.use_l2_conv_refine else nn.Identity()
        self.interface_vec = nn.Sequential(nn.LayerNorm(D * 2 + 1), nn.Linear(D * 2 + 1, D), nn.GELU(), nn.Dropout(cfg.dropout))
        self.interface_logit = nn.Linear(D, 1)

        self.attn_pool = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.struct_summary = StructureSummary(D) if cfg.use_structure_summary else None
        pair_in = D * 8 if cfg.use_structure_summary else D * 4
        self.pair_vec = nn.Sequential(nn.LayerNorm(pair_in), nn.Linear(pair_in, D), nn.GELU(), nn.Dropout(cfg.dropout))
        self.pair_logit = nn.Linear(D, 1)

        self.gate = nn.Sequential(nn.LayerNorm(D * 3), nn.Linear(D * 3, D), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(D, 3))
        self.reliability_head = nn.Sequential(nn.LayerNorm(D * 3), nn.Linear(D * 3, D), nn.GELU(), nn.Linear(D, 1))

    @staticmethod
    def project_interface_to_residue(S_interface, maskA, maskB):
        return project_interface_to_residue(S_interface, maskA, maskB)

    def _attention_pool(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        a = self.attn_pool(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~_bmask(mask), -1e9)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)

    @staticmethod
    def _topk_evidence_pool(
        x: torch.Tensor,
        logits: torch.Tensor,
        idx: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        top_feat = x.gather(1, gather_idx)
        top_logits = logits.gather(1, idx)
        if mask is not None:
            top_valid = _bmask(mask).gather(1, idx)
            top_logits = top_logits.masked_fill(~top_valid, -1e9)
        all_bad = top_logits.le(-1e8).all(dim=1, keepdim=True)
        attn_logits = top_logits.masked_fill(all_bad, 0.0)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        z_topk = (top_feat * attn).sum(dim=1)
        fallback = _masked_mean(x * torch.sigmoid(logits).unsqueeze(-1), mask)
        z_topk = torch.where(all_bad.expand(-1, x.size(-1)), fallback, z_topk)
        safe_logits = top_logits.masked_fill(top_logits.le(-1e8), 0.0)
        mean = safe_logits.mean(dim=1, keepdim=True)
        maxv = safe_logits.max(dim=1, keepdim=True).values
        std = safe_logits.std(dim=1, unbiased=False, keepdim=True)
        stats = torch.cat([mean, maxv, std], dim=-1)
        return z_topk, stats

    def _geometry_bias(
        self,
        coordsA: Optional[torch.Tensor],
        coordsB: Optional[torch.Tensor],
        maskA: torch.Tensor,
        maskB: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if coordsA is None or coordsB is None:
            return None
        if coordsA.dim() == 2:
            coordsA = coordsA.unsqueeze(0)
        if coordsB.dim() == 2:
            coordsB = coordsB.unsqueeze(0)
        if (
            coordsA.size(0) != maskA.size(0)
            or coordsB.size(0) != maskB.size(0)
            or coordsA.size(1) != maskA.size(1)
            or coordsB.size(1) != maskB.size(1)
        ):
            return None
        finiteA = torch.isfinite(coordsA).all(dim=-1) & _bmask(maskA)
        finiteB = torch.isfinite(coordsB).all(dim=-1) & _bmask(maskB)
        coordsA = torch.nan_to_num(coordsA.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coordsB = torch.nan_to_num(coordsB.float(), nan=0.0, posinf=0.0, neginf=0.0)
        dist = torch.cdist(coordsA, coordsB)
        tau = max(float(self.cfg.l2_geom_tau), 1e-3)
        bias = torch.exp(-dist / tau).to(device=coordsA.device)
        valid = finiteA.unsqueeze(2) & finiteB.unsqueeze(1)
        return bias.masked_fill(~valid, 0.0)

    @staticmethod
    def _l1_geometry_features(coords: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if coords is None:
            return None
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        valid = torch.isfinite(coords).all(dim=-1)
        if mask is not None:
            valid = valid & _bmask(mask)
        if not bool(valid.any()):
            return None
        c = torch.nan_to_num(coords.float(), nan=0.0, posinf=0.0, neginf=0.0)
        dist = torch.cdist(c, c).clamp_min(0.0)
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
        eye = torch.eye(dist.size(1), dtype=torch.bool, device=dist.device).unsqueeze(0)
        pair_valid = pair_valid & ~eye
        far = torch.full_like(dist, 1e4)
        dist_valid = torch.where(pair_valid, dist, far)
        density8 = ((dist < 8.0) & pair_valid).sum(dim=-1).float() / 20.0
        density12 = ((dist < 12.0) & pair_valid).sum(dim=-1).float() / 35.0
        k = min(8, dist.size(1))
        near = torch.topk(dist_valid, k=k, dim=-1, largest=False).values
        near = torch.where(near < 1e3, near, torch.zeros_like(near))
        near_count = (near > 0).sum(dim=-1).float().clamp_min(1.0)
        near_mean = near.sum(dim=-1) / near_count / 12.0
        near_min = torch.where(dist_valid.min(dim=-1).values < 1e3, dist_valid.min(dim=-1).values, torch.zeros_like(density8)) / 8.0
        center = (c * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        radial = (c - center).norm(dim=-1)
        radial = radial / radial.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True).clamp_min(1.0)
        rel_pos = torch.linspace(0.0, 1.0, c.size(1), device=c.device, dtype=c.dtype).unsqueeze(0).expand(c.size(0), -1)
        feat = torch.stack([density8, density12, near_mean, near_min, radial, rel_pos], dim=-1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.masked_fill(~valid.unsqueeze(-1), 0.0)

    def forward(
        self,
        resA: torch.Tensor,
        maskA: Optional[torch.Tensor] = None,
        chainA: Optional[torch.Tensor] = None,
        coordsA: Optional[torch.Tensor] = None,
        resB: Optional[torch.Tensor] = None,
        maskB: Optional[torch.Tensor] = None,
        chainB: Optional[torch.Tensor] = None,
        coordsB: Optional[torch.Tensor] = None,
        l2_ablate_topm_support: bool = False,
        l2_ablate_endpoint_sparse: bool = False,
        **unused,
    ) -> Dict:
        if resB is None:
            raise ValueError("TRIAGEPPIModel.forward requires resB.")
        if resA.dim() == 2:
            resA = resA.unsqueeze(0)
        if resB.dim() == 2:
            resB = resB.unsqueeze(0)
        maskA = _bmask(maskA) if maskA is not None else torch.ones(resA.shape[:2], dtype=torch.bool, device=resA.device)
        maskB = _bmask(maskB) if maskB is not None else torch.ones(resB.shape[:2], dtype=torch.bool, device=resB.device)

        single_chain_l1 = bool(getattr(self.cfg, "l1_single_chain_mode", False)) and resB.size(1) <= 1
        xA = self.filmA(self.projA(torch.nan_to_num(resA.float())), chainA)
        if single_chain_l1 and bool(getattr(self.cfg, "use_l1_geom_early", False)):
            geom_feat_A = self._l1_geometry_features(coordsA, maskA)
            if geom_feat_A is not None and geom_feat_A.size(1) == xA.size(1):
                geom_emb_A = self.l1_geom_embed_A(geom_feat_A.to(dtype=xA.dtype, device=xA.device))
                xA = xA + float(getattr(self.cfg, "l1_geom_early_alpha", 0.25)) * geom_emb_A
        HA = self.encA(xA, maskA)
        HB = self.encB(self.filmB(self.projB(torch.nan_to_num(resB.float())), chainB), maskB)
        if not single_chain_l1:
            HA, HB = self.cross(HA, HB, maskA, maskB)

        logit_res_A = self.res_head_A(HA, maskA).clamp(-30.0, 30.0)
        if single_chain_l1 and bool(getattr(self.cfg, "use_l1_raw_skip", False)):
            raw_logit_A = self.l1_raw_head_A(torch.nan_to_num(resA.float())).squeeze(-1).clamp(-30.0, 30.0)
            logit_res_A = (logit_res_A + float(getattr(self.cfg, "l1_raw_skip_alpha", 0.50)) * raw_logit_A).clamp(-30.0, 30.0)
        if single_chain_l1 and bool(getattr(self.cfg, "use_l1_geom_adapter", False)):
            geom_feat_A = self._l1_geometry_features(coordsA, maskA)
            if geom_feat_A is not None and geom_feat_A.size(1) == logit_res_A.size(1):
                geom_logit_A = self.l1_geom_adapter_A(geom_feat_A.to(dtype=logit_res_A.dtype, device=logit_res_A.device)).squeeze(-1)
                logit_res_A = (logit_res_A + float(getattr(self.cfg, "l1_geom_alpha", 0.15)) * geom_logit_A).clamp(-30.0, 30.0)
        logit_res_B = self.res_head_B(HB, maskB).clamp(-30.0, 30.0)
        p_res_A = torch.sigmoid(logit_res_A)
        p_res_B = torch.sigmoid(logit_res_B)
        topk_val_A, topk_res_A_idx = select_topk_residues(logit_res_A, maskA, self.cfg.topk, self.cfg.block_size, self.cfg.block_topr)
        topk_val_B, topk_res_B_idx = select_topk_residues(logit_res_B, maskB, self.cfg.topk, self.cfg.block_size, self.cfg.block_topr)

        zA_topk, statsA_topk = self._topk_evidence_pool(HA, logit_res_A, topk_res_A_idx, maskA)
        zB_topk, statsB_topk = self._topk_evidence_pool(HB, logit_res_B, topk_res_B_idx, maskB)
        z_res = self.res_vec(torch.cat([zA_topk, zB_topk, statsA_topk, statsB_topk], dim=-1))
        logit_res = self.res_logit(z_res).squeeze(-1)

        PA = F.normalize(self.l2_projA(HA), dim=-1)
        PB = F.normalize(self.l2_projB(HB), dim=-1)
        S_raw = torch.einsum("bid,bjd->bij", PA, PB) / math.sqrt(max(1, self.cfg.d_pair))
        S_interface = S_raw + self.cfg.l2_gate_alpha * (p_res_A.unsqueeze(2) + p_res_B.unsqueeze(1))
        valid2d = maskA.unsqueeze(2) & maskB.unsqueeze(1)
        geom_bias = self._geometry_bias(coordsA, coordsB, maskA, maskB)
        if geom_bias is not None:
            beta = self.l2_geom_beta.clamp(0.0, 2.0)
            S_interface = S_interface + beta.to(dtype=S_interface.dtype) * geom_bias.to(dtype=S_interface.dtype, device=S_interface.device)
        S_interface = self.l2_refine(S_interface.masked_fill(~valid2d, 0.0)).masked_fill(~valid2d, -30.0).clamp(-30.0, 30.0)
        topm_interface_idx, topm_interface_logits = select_topm_interface(S_interface, maskA, maskB, self.cfg.topm, self.cfg.block_size)
        top_i, top_j = topm_interface_idx[..., 0], topm_interface_idx[..., 1]
        featA = HA.gather(1, top_i.unsqueeze(-1).expand(-1, -1, HA.size(-1)))
        featB = HB.gather(1, top_j.unsqueeze(-1).expand(-1, -1, HB.size(-1)))
        if l2_ablate_endpoint_sparse:
            featA = torch.zeros_like(featA)
            featB = torch.zeros_like(featB)
        if l2_ablate_topm_support:
            featA = torch.zeros_like(featA)
            featB = torch.zeros_like(featB)
            topm_interface_logits = torch.full_like(topm_interface_logits, -30.0)
        pair_sparse_feat = torch.cat([featA, featB, topm_interface_logits.unsqueeze(-1)], dim=-1)
        sparse_w = torch.softmax(topm_interface_logits, dim=1).unsqueeze(-1)
        z_interface = self.interface_vec((pair_sparse_feat * sparse_w).sum(dim=1))
        logit_interface = self.interface_logit(z_interface).squeeze(-1)

        poolA = self._attention_pool(HA, maskA)
        poolB = self._attention_pool(HB, maskB)
        pair_parts = [poolA, poolB, torch.abs(poolA - poolB), poolA * poolB]
        if self.struct_summary is not None:
            sA = self.struct_summary(coordsA, maskA)
            sB = self.struct_summary(coordsB, maskB)
            if sA is None:
                sA = torch.zeros_like(poolA)
            if sB is None:
                sB = torch.zeros_like(poolB)
            pair_parts.extend([sA, sB, torch.abs(sA - sB), sA * sB])
        z_pair = self.pair_vec(torch.cat(pair_parts, dim=-1))
        logit_pair_raw = self.pair_logit(z_pair).squeeze(-1)
        p_pair_raw = torch.sigmoid(logit_pair_raw)

        evidence = torch.cat([z_res, z_interface, z_pair], dim=-1)
        gates = torch.softmax(self.gate(evidence), dim=-1)
        g_res, g_interface, g_pair = gates[:, 0], gates[:, 1], gates[:, 2]
        logit_triage = g_res * logit_res + g_interface * logit_interface + g_pair * logit_pair_raw
        p_triage = torch.sigmoid(logit_triage)
        evidence_reliability = torch.sigmoid(self.reliability_head(evidence).squeeze(-1))

        return {
            "p_triage": p_triage,
            "logit_triage": logit_triage,
            "p_pair_raw": p_pair_raw,
            "logit_pair_raw": logit_pair_raw,
            "p_res_A": p_res_A,
            "p_res_B": p_res_B,
            "logit_res_A": logit_res_A,
            "logit_res_B": logit_res_B,
            "S_interface": S_interface,
            "p_interface": torch.sigmoid(S_interface),
            "topk_res_A_idx": topk_res_A_idx,
            "topk_res_B_idx": topk_res_B_idx,
            "topm_interface_idx": topm_interface_idx,
            "topm_interface_logits": topm_interface_logits,
            "topm_interface_prob": torch.sigmoid(topm_interface_logits),
            "z_res": z_res,
            "z_interface": z_interface,
            "z_pair": z_pair,
            "fusion_weights": {
                "g_res": g_res,
                "g_interface": g_interface,
                "g_pair": g_pair,
            },
            "evidence_reliability": evidence_reliability,
            "logit_res": logit_res,
            "logit_interface": logit_interface,
        }
