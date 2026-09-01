"""Training objectives described in the PertiNet manuscript."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5, embed_dim=64):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.seq_proj = nn.Linear(64, embed_dim)
        self.struct_proj = nn.Linear(128, embed_dim)
        self.go_proj = nn.Linear(64, embed_dim)

    def forward(self, h_seq, h_struct, h_go):
        anchor = self.seq_proj(h_seq)
        positive = self.struct_proj(h_struct)
        negative1 = self.go_proj(h_go)
        negative2 = negative1[torch.randperm(negative1.size(0), device=negative1.device)]
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative1)
        d_nn = F.pairwise_distance(negative1, negative2)
        return (
            F.relu(d_ap - d_an + self.margin1)
            + F.relu(d_ap - d_nn + self.margin2)
        ).mean()


class PertiNetObjective(nn.Module):
    """Weighted BCE + quadruplet loss + score-separation regularizer."""

    def __init__(self, pos_weight=1.0, lambda_quad=0.1, lambda_sep=0.2):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.lambda_quad = float(lambda_quad)
        self.lambda_sep = float(lambda_sep)
        self.quadruplet = QuadrupletLoss()

    def forward(self, outputs, labels):
        logits, score_separation, _, h_seq, h_struct, h_go, _ = outputs
        bce = F.binary_cross_entropy_with_logits(
            logits.view(-1), labels.float().view(-1), pos_weight=self.pos_weight
        )
        quad = self.quadruplet(h_seq, h_struct, h_go)
        sep = -score_separation.mean()
        total = bce + self.lambda_quad * quad + self.lambda_sep * sep
        return total, {"bce": bce.detach(), "quad": quad.detach(), "sep": sep.detach()}
