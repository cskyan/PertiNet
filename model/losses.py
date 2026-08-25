# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from model import TRIAGEConfig, project_interface_to_residue


def _zero_like(out: Dict) -> torch.Tensor:
    return out["logit_triage"].new_zeros(())


def _masked_bce_logits(logits: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    if mask is not None:
        mask = mask.to(device=logits.device).bool()
        if mask.sum() == 0:
            return logits.new_zeros(())
        logits = logits[mask]
        target = target[mask]
    return F.binary_cross_entropy_with_logits(logits, target)


def _smooth_target(target: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    smoothing = float(smoothing)
    if smoothing <= 0:
        return target
    return target * (1.0 - smoothing) + 0.5 * smoothing


def _masked_bce_logits_balanced(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    cap: float = 12.0,
    smoothing: float = 0.0,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    if mask is not None:
        mask = mask.to(device=logits.device).bool()
        if mask.sum() == 0:
            return logits.new_zeros(())
        logits = logits[mask]
        target = target[mask]
    pos = target.sum()
    neg = target.numel() - pos
    target = _smooth_target(target, smoothing)
    if pos <= 0:
        return F.binary_cross_entropy_with_logits(logits, target)
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(cap)).detach()
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def _l1_per_protein_bce_logits_balanced(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    cap: float = 12.0,
    extreme_weight: float = 1.0,
    zero_label_weight: float = -1.0,
    full_label_weight: float = -1.0,
    smoothing: float = 0.0,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = torch.ones_like(target, dtype=torch.bool, device=logits.device)
    if mask is not None:
        valid = mask.to(device=logits.device).bool()
    losses = []
    for b in range(logits.size(0)):
        m = valid[b]
        if m.sum() == 0:
            continue
        s = logits[b][m]
        y = target[b][m]
        pos = y.sum()
        neg = y.numel() - pos
        ys = _smooth_target(y, smoothing)
        if pos > 0:
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(cap)).detach()
            loss = F.binary_cross_entropy_with_logits(s, ys, pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(s, ys)
        if pos <= 0:
            weight = float(zero_label_weight) if float(zero_label_weight) >= 0 else float(extreme_weight)
            loss = loss * weight
        elif neg <= 0:
            weight = float(full_label_weight) if float(full_label_weight) >= 0 else float(extreme_weight)
            loss = loss * weight
        losses.append(loss)
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _masked_pairwise_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.20,
    max_pairs: int = 2048,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    if mask is not None:
        mask = mask.to(device=logits.device).bool()
        if mask.sum() == 0:
            return logits.new_zeros(())
        logits = logits[mask]
        target = target[mask]
    pos = logits[target > 0.5]
    neg = logits[target <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    if pos.numel() > max_pairs:
        pos = pos[torch.linspace(0, pos.numel() - 1, max_pairs, device=pos.device).long()]
    if neg.numel() > max_pairs:
        neg = neg[torch.linspace(0, neg.numel() - 1, max_pairs, device=neg.device).long()]
    diff = pos[:, None] - neg[None, :]
    return F.relu(float(margin) - diff).mean()


def _l1_hard_rank_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 0.50,
    neg_frac: float = 0.20,
    max_neg: int = 128,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = torch.ones_like(target, dtype=torch.bool, device=logits.device)
    if mask is not None:
        valid = mask.to(device=logits.device).bool()
    losses = []
    for b in range(logits.size(0)):
        m = valid[b]
        if m.sum() <= 1:
            continue
        s = logits[b][m]
        y = target[b][m]
        pos = s[y > 0.5]
        neg = s[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        k = max(1, int(round(float(neg_frac) * float(neg.numel()))))
        k = min(k, int(max_neg), int(neg.numel()))
        hard_neg = torch.topk(neg, k, largest=True).values
        losses.append(F.relu(hard_neg[None, :] + float(margin) - pos[:, None]).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _l1_topband_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    frac: float = 0.20,
    min_k: int = 10,
    max_k: int = 128,
    cap: float = 12.0,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = torch.ones_like(target, dtype=torch.bool, device=logits.device)
    if mask is not None:
        valid = mask.to(device=logits.device).bool()
    losses = []
    for b in range(logits.size(0)):
        m = valid[b]
        L = int(m.sum().item())
        if L <= 1:
            continue
        s = logits[b][m]
        y = target[b][m]
        k = max(int(min_k), int(round(float(frac) * float(L))))
        k = min(k, int(max_k), L)
        idx = torch.topk(s.detach(), k, largest=True).indices
        yy = y[idx]
        ss = s[idx]
        pos = yy.sum()
        if pos > 0:
            neg = yy.numel() - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, float(cap)).detach()
            losses.append(F.binary_cross_entropy_with_logits(ss, yy, pos_weight=pos_weight))
        else:
            losses.append(F.binary_cross_entropy_with_logits(ss, yy))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _l1_l10_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    frac: float = 0.10,
    margin: float = 0.04,
    max_pos: int = 64,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = torch.ones_like(target, dtype=torch.bool, device=logits.device)
    if mask is not None:
        valid = mask.to(device=logits.device).bool()
    losses = []
    for b in range(logits.size(0)):
        m = valid[b]
        L = int(m.sum().item())
        if L <= 1:
            continue
        s = logits[b][m]
        y = target[b][m]
        pos_mask = y > 0.5
        n_pos = int(pos_mask.sum().item())
        n_neg = int((~pos_mask).sum().item())
        if n_pos <= 0 or n_neg <= 0:
            continue
        k = max(1, int(torch.ceil(s.new_tensor(float(L) * float(frac))).item()))
        k = min(k, L)
        boundary = torch.topk(s.detach(), k, largest=True).values[-1]
        pos_scores = s[pos_mask]
        pos_losses = F.relu(boundary + float(margin) - pos_scores)
        pos_losses = pos_losses[pos_losses > 0]
        if pos_losses.numel() == 0:
            continue
        if pos_losses.numel() > int(max_pos):
            pos_losses = torch.topk(pos_losses, int(max_pos), largest=True).values
        losses.append(pos_losses.mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def _epoch_ramp_weight(base: float, epoch: int, start_epoch: int = 1, ramp_epochs: int = 1) -> float:
    base = float(base)
    if base <= 0:
        return 0.0
    start_epoch = int(start_epoch)
    ramp_epochs = max(1, int(ramp_epochs))
    if int(epoch) < start_epoch:
        return 0.0
    return base * min(1.0, float(int(epoch) - start_epoch + 1) / float(ramp_epochs))


def _batch_pair_label(batch: Dict, out: Dict) -> torch.Tensor:
    for key in ("y_pair", "label", "labels", "has_contact", "target"):
        if key in batch:
            return batch[key].to(device=out["logit_triage"].device, dtype=out["logit_triage"].dtype).view(-1)
    raise KeyError("PAIR batch needs one of y_pair/label/labels/has_contact/target.")


def _struct_pair_label(batch: Dict, out: Dict) -> torch.Tensor:
    if "y_struct" in batch:
        return batch["y_struct"].to(device=out["logit_triage"].device, dtype=out["logit_triage"].dtype).view(-1)
    if "y2d" not in batch:
        return _batch_pair_label(batch, out)
    y2d = batch["y2d"].to(device=out["logit_triage"].device, dtype=out["logit_triage"].dtype)
    valid = torch.ones_like(y2d, dtype=torch.bool)
    if "maskA" in batch:
        valid = valid & (batch["maskA"].to(y2d.device) > 0.5).unsqueeze(2)
    if "maskB" in batch:
        valid = valid & (batch["maskB"].to(y2d.device) > 0.5).unsqueeze(1)
    return ((y2d > 0.5) & valid).flatten(1).any(dim=1).to(dtype=out["logit_triage"].dtype)


def _topm_ranking_loss(out: Dict, y2d: torch.Tensor) -> torch.Tensor:
    idx = out["topm_interface_idx"]
    logits = out["topm_interface_logits"]
    B, M, _ = idx.shape
    labels = []
    for b in range(B):
        labels.append(y2d[b, idx[b, :, 0], idx[b, :, 1]])
    y = torch.stack(labels, dim=0).to(device=logits.device, dtype=logits.dtype)
    if y.sum() <= 0:
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits, y)


def _topk_margin_rank_loss(
    scores: torch.Tensor,
    y2d: torch.Tensor,
    valid2d: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses = []
    for b in range(scores.size(0)):
        valid = valid2d[b]
        pos = valid & (y2d[b] > 0.5)
        neg = valid & (y2d[b] <= 0.5)
        if pos.any() and neg.any():
            pos_boundary = scores[b][pos].max()
            neg_boundary = scores[b][neg].max()
            losses.append(F.relu(neg_boundary + float(margin) - pos_boundary))
    if not losses:
        return scores.new_zeros(())
    return torch.stack(losses).mean()


def compute_struct_loss(out: Dict, batch: Dict, cfg: TRIAGEConfig, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = _zero_like(out)
    logs: Dict[str, float] = {}
    device = out["logit_triage"].device

    l1_only = "y2d" not in batch and "y_pair" not in batch and "y_struct" not in batch

    if "y_res_A" in batch:
        if l1_only or bool(getattr(cfg, "dest_balanced_res_loss", False)):
            if bool(getattr(cfg, "l1_per_protein_loss", False)):
                loss = _l1_per_protein_bce_logits_balanced(
                    out["logit_res_A"],
                    batch["y_res_A"],
                    batch.get("maskA"),
                    cap=getattr(cfg, "l1_pos_weight_cap", 12.0),
                    extreme_weight=getattr(cfg, "l1_extreme_label_weight", 1.0),
                    zero_label_weight=getattr(cfg, "l1_zero_label_weight", -1.0),
                    full_label_weight=getattr(cfg, "l1_full_label_weight", -1.0),
                    smoothing=getattr(cfg, "l1_label_smoothing", 0.0),
                )
            else:
                loss = _masked_bce_logits_balanced(
                    out["logit_res_A"],
                    batch["y_res_A"],
                    batch.get("maskA"),
                    cap=getattr(cfg, "l1_pos_weight_cap", 12.0),
                    smoothing=getattr(cfg, "l1_label_smoothing", 0.0),
                )
        else:
            loss = _masked_bce_logits(out["logit_res_A"], batch["y_res_A"], batch.get("maskA"))
        total = total + cfg.w_res * loss
        logs["L_res_A"] = float(loss.detach().cpu())
        rank_w = _epoch_ramp_weight(
            getattr(cfg, "w_l1_rank", 0.0),
            epoch,
            getattr(cfg, "l1_rank_start_epoch", 1),
            getattr(cfg, "l1_rank_ramp_epochs", 1),
        )
        if l1_only and rank_w > 0:
            rank = _masked_pairwise_rank_loss(
                out["logit_res_A"],
                batch["y_res_A"],
                batch.get("maskA"),
                margin=getattr(cfg, "l1_rank_margin", 0.20),
                max_pairs=getattr(cfg, "l1_rank_max_pairs", 2048),
            )
            total = total + rank_w * rank
            logs["L_l1_rank_A"] = float(rank.detach().cpu())
            logs["w_l1_rank"] = float(rank_w)
        hard_rank_w = _epoch_ramp_weight(
            getattr(cfg, "w_l1_hard_rank", 0.0),
            epoch,
            getattr(cfg, "l1_hard_rank_start_epoch", 1),
            getattr(cfg, "l1_hard_rank_ramp_epochs", 1),
        )
        if l1_only and hard_rank_w > 0:
            hard_rank = _l1_hard_rank_loss(
                out["logit_res_A"],
                batch["y_res_A"],
                batch.get("maskA"),
                margin=getattr(cfg, "l1_hard_rank_margin", 0.50),
                neg_frac=getattr(cfg, "l1_hard_rank_neg_frac", 0.20),
                max_neg=getattr(cfg, "l1_hard_rank_max_neg", 128),
            )
            total = total + hard_rank_w * hard_rank
            logs["L_l1_hard_rank_A"] = float(hard_rank.detach().cpu())
            logs["w_l1_hard_rank"] = float(hard_rank_w)
        topband_w = _epoch_ramp_weight(
            getattr(cfg, "w_l1_topband_bce", 0.0),
            epoch,
            getattr(cfg, "l1_topband_start_epoch", 1),
            getattr(cfg, "l1_topband_ramp_epochs", 1),
        )
        if l1_only and topband_w > 0:
            topband = _l1_topband_bce_loss(
                out["logit_res_A"],
                batch["y_res_A"],
                batch.get("maskA"),
                frac=getattr(cfg, "l1_topband_frac", 0.20),
                min_k=getattr(cfg, "l1_topband_min_k", 10),
                max_k=getattr(cfg, "l1_topband_max_k", 128),
                cap=getattr(cfg, "l1_pos_weight_cap", 12.0),
            )
            total = total + topband_w * topband
            logs["L_l1_topband_A"] = float(topband.detach().cpu())
            logs["w_l1_topband"] = float(topband_w)
        l10_boundary_w = _epoch_ramp_weight(
            getattr(cfg, "w_l1_l10_boundary", 0.0),
            epoch,
            getattr(cfg, "l1_l10_boundary_start_epoch", 1),
            getattr(cfg, "l1_l10_boundary_ramp_epochs", 1),
        )
        if l1_only and l10_boundary_w > 0:
            l10_boundary = _l1_l10_boundary_loss(
                out["logit_res_A"],
                batch["y_res_A"],
                batch.get("maskA"),
                frac=getattr(cfg, "l1_l10_boundary_frac", 0.10),
                margin=getattr(cfg, "l1_l10_boundary_margin", 0.04),
                max_pos=getattr(cfg, "l1_l10_boundary_max_pos", 64),
            )
            total = total + l10_boundary_w * l10_boundary
            logs["L_l1_l10_boundary_A"] = float(l10_boundary.detach().cpu())
            logs["w_l1_l10_boundary"] = float(l10_boundary_w)
    if "y_res_B" in batch:
        if l1_only or bool(getattr(cfg, "dest_balanced_res_loss", False)):
            if bool(getattr(cfg, "l1_per_protein_loss", False)):
                loss = _l1_per_protein_bce_logits_balanced(
                    out["logit_res_B"],
                    batch["y_res_B"],
                    batch.get("maskB"),
                    cap=getattr(cfg, "l1_pos_weight_cap", 12.0),
                    extreme_weight=getattr(cfg, "l1_extreme_label_weight", 1.0),
                    zero_label_weight=getattr(cfg, "l1_zero_label_weight", -1.0),
                    full_label_weight=getattr(cfg, "l1_full_label_weight", -1.0),
                    smoothing=getattr(cfg, "l1_label_smoothing", 0.0),
                )
            else:
                loss = _masked_bce_logits_balanced(
                    out["logit_res_B"],
                    batch["y_res_B"],
                    batch.get("maskB"),
                    cap=getattr(cfg, "l1_pos_weight_cap", 12.0),
                    smoothing=getattr(cfg, "l1_label_smoothing", 0.0),
                )
        else:
            loss = _masked_bce_logits(out["logit_res_B"], batch["y_res_B"], batch.get("maskB"))
        total = total + cfg.w_res * loss
        logs["L_res_B"] = float(loss.detach().cpu())
        rank_w = _epoch_ramp_weight(
            getattr(cfg, "w_l1_rank", 0.0),
            epoch,
            getattr(cfg, "l1_rank_start_epoch", 1),
            getattr(cfg, "l1_rank_ramp_epochs", 1),
        )
        if l1_only and rank_w > 0:
            rank = _masked_pairwise_rank_loss(
                out["logit_res_B"],
                batch["y_res_B"],
                batch.get("maskB"),
                margin=getattr(cfg, "l1_rank_margin", 0.20),
                max_pairs=getattr(cfg, "l1_rank_max_pairs", 2048),
            )
            total = total + rank_w * rank
            logs["L_l1_rank_B"] = float(rank.detach().cpu())
            logs["w_l1_rank"] = float(rank_w)

    if "y2d" in batch:
        y2d = batch["y2d"].to(device=device, dtype=out["S_interface"].dtype)
        valid2d = torch.ones_like(y2d, dtype=torch.bool)
        if "maskA" in batch:
            valid2d = valid2d & (batch["maskA"].to(device) > 0.5).unsqueeze(2)
        if "maskB" in batch:
            valid2d = valid2d & (batch["maskB"].to(device) > 0.5).unsqueeze(1)
        loss = _masked_bce_logits(out["S_interface"], y2d, valid2d)
        total = total + cfg.w_contact * loss
        logs["L_contact_2D"] = float(loss.detach().cpu())

        projA, projB = project_interface_to_residue(out["S_interface"], batch.get("maskA"), batch.get("maskB"))
        cons = _zero_like(out)
        n = 0
        if "y_res_A" in batch:
            cons = cons + F.mse_loss(out["p_res_A"], projA.detach())
            n += 1
        if "y_res_B" in batch:
            cons = cons + F.mse_loss(out["p_res_B"], projB.detach())
            n += 1
        if n:
            cons = cons / n
            total = total + cfg.w_l1_l2_consistency * cons
            logs["L_l1_l2_consistency"] = float(cons.detach().cpu())

        rank = _topm_ranking_loss(out, y2d)
        total = total + cfg.w_topk_contact_ranking * rank
        logs["L_topk_contact_ranking"] = float(rank.detach().cpu())

        margin_rank = _topk_margin_rank_loss(
            out["S_interface"],
            y2d,
            valid2d,
            getattr(cfg, "topk_rank_margin", 0.50),
        )
        total = total + getattr(cfg, "w_topk_margin_rank", 0.0) * margin_rank
        logs["L_topk_margin_rank"] = float(margin_rank.detach().cpu())

    if "y2d" in batch or "y_pair" in batch or "y_struct" in batch:
        y_struct = _struct_pair_label(batch, out)
        tri = F.binary_cross_entropy_with_logits(out["logit_triage"], y_struct)
        total = total + cfg.w_triage_struct * tri
        logs["L_triage_struct"] = float(tri.detach().cpu())

    g_res = out["fusion_weights"]["g_res"]
    g_interface = out["fusion_weights"]["g_interface"]
    g_pair = out["fusion_weights"]["g_pair"]
    local_gate = g_res + g_interface
    struct_gate = (
        F.relu(g_pair - float(getattr(cfg, "struct_max_pair_gate", 0.45))).mean()
        + F.relu(float(getattr(cfg, "struct_min_local_gate", 0.35)) - local_gate).mean()
    )
    total = total + float(getattr(cfg, "w_struct_gate_regularization", 0.0)) * struct_gate
    logs["L_struct_gate_regularization"] = float(struct_gate.detach().cpu())
    logs["loss"] = float(total.detach().cpu())
    return total, logs


def compute_pair_loss(out: Dict, batch: Dict, cfg: TRIAGEConfig, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    del epoch
    y = _batch_pair_label(batch, out)
    total = F.binary_cross_entropy_with_logits(out["logit_triage"], y) * cfg.w_triage_pair
    logs = {"L_triage_pair": float((total / max(cfg.w_triage_pair, 1e-8)).detach().cpu())}

    topm_logits = out["topm_interface_logits"]
    mil_logit = torch.logsumexp(topm_logits, dim=1) - torch.log(torch.tensor(topm_logits.size(1), device=topm_logits.device, dtype=topm_logits.dtype))
    mil = F.binary_cross_entropy_with_logits(mil_logit, y)
    total = total + cfg.w_l2_mil_weak * mil
    logs["L_l2_mil_weak"] = float(mil.detach().cpu())

    cons = F.mse_loss(torch.sigmoid(mil_logit), out["p_triage"].detach())
    total = total + cfg.w_interface_pair_consistency * cons
    logs["L_interface_pair_consistency"] = float(cons.detach().cpu())

    neg = y <= 0.5
    if neg.any():
        suppress = out["topm_interface_prob"][neg].mean() + out["p_res_A"][neg].mean() + out["p_res_B"][neg].mean()
        suppress = suppress / 3.0
    else:
        suppress = out["logit_triage"].new_zeros(())
    total = total + cfg.w_negative_evidence_suppression * suppress
    logs["L_negative_evidence_suppression"] = float(suppress.detach().cpu())

    gates = torch.stack([
        out["fusion_weights"]["g_res"],
        out["fusion_weights"]["g_interface"],
        out["fusion_weights"]["g_pair"],
    ], dim=-1)
    entropy = -(gates * gates.clamp_min(1e-8).log()).sum(dim=-1).mean()
    pair_dom = F.relu(gates[:, 2] - float(getattr(cfg, "max_pair_gate", 0.85))).mean()
    low_entropy = F.relu(float(getattr(cfg, "min_gate_entropy", 0.80)) - entropy).mean()
    res_under = F.relu(float(getattr(cfg, "min_res_gate", 0.05)) - gates[:, 0]).mean()
    gate_reg = pair_dom + 0.1 * low_entropy + 0.5 * res_under
    total = total + cfg.w_gate_regularization * gate_reg
    logs["L_gate_regularization"] = float(gate_reg.detach().cpu())
    logs["L_gate_res_under"] = float(res_under.detach().cpu())

    rel_w = float(getattr(cfg, "w_reliability", 0.0))
    if rel_w > 0 and "evidence_reliability" in out:
        p_fuse = out["p_triage"].detach()
        y_rel = y.to(device=p_fuse.device, dtype=p_fuse.dtype).view_as(p_fuse)
        p_res = torch.sigmoid(out["logit_res"].detach())
        p_interface = torch.sigmoid(out["logit_interface"].detach())
        p_pair = torch.sigmoid(out["logit_pair_raw"].detach())
        c_fuse = 1.0 - (p_fuse - y_rel).abs()
        a_branch = 1.0 - torch.stack(
            [
                (p_res - p_fuse).abs(),
                (p_interface - p_fuse).abs(),
                (p_pair - p_fuse).abs(),
            ],
            dim=-1,
        ).mean(dim=-1)
        t_rel = (c_fuse * a_branch).clamp(0.0, 1.0).detach()
        l_rel = F.binary_cross_entropy(out["evidence_reliability"].view_as(t_rel), t_rel)
        total = total + rel_w * l_rel
        logs["L_reliability"] = float(l_rel.detach().cpu())
    logs["loss"] = float(total.detach().cpu())
    return total, logs


def compute_joint_loss(out: Dict, batch: Dict, task: str, cfg: TRIAGEConfig, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    task_l = str(task).lower()
    if task_l in ("struct", "structure", "drn", "dips", "l1", "graphrbf"):
        return compute_struct_loss(out, batch, cfg, epoch)
    if task_l in ("pair", "tuna", "huri", "bioplex"):
        return compute_pair_loss(out, batch, cfg, epoch)
    raise ValueError(f"Unknown TRIAGE task: {task}")
