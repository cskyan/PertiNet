#!/usr/bin/env python3
"""Train a case-safe symmetric ESM pair model with validation-only calibration."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SEEDS = (1337, 2027, 3407, 4517, 5651)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-root", type=Path, required=True)
    p.add_argument("--esm-cache", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--architecture", choices=("siamese_compact", "siamese_gated", "pair_mlp"),
                   default="siamese_compact")
    p.add_argument("--pooling", choices=("mean_max", "mean_max_std", "mean_max_std_topk",
                                          "mean_max_std_topk_segments"), default="mean_max_std")
    p.add_argument("--grad-clip", type=float, default=2.0)
    p.add_argument("--sampling", choices=("balanced", "weighted_loss"), default="balanced",
                   help="Balanced sampler is recommended for the 90/10 V2 training split.")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--evaluate-test", action="store_true",
                   help="Evaluate test.tsv. Keep OFF during model development; use only for a locked final run.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_embedding(cache: Path, protein: str, pooling="mean_max"):
    path = cache / f"{protein}_esm.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("representations", "embedding", "embeddings"):
            if key in value:
                value = value[key]
                if isinstance(value, dict): value = value[max(value)]
                break
    value = torch.as_tensor(value).float()
    if value.ndim == 3 and value.shape[0] == 1: value = value[0]
    if value.ndim != 2 or value.shape[1] < 1280:
        raise ValueError(f"Bad ESM tensor for {protein}: {tuple(value.shape)}")
    value = value[:, :1280]
    parts = [value.mean(0), value.amax(0)]
    if pooling in ("mean_max_std", "mean_max_std_topk", "mean_max_std_topk_segments"):
        parts.append(value.std(0, unbiased=False))
    if pooling in ("mean_max_std_topk", "mean_max_std_topk_segments"):
        k=max(1,int(np.ceil(value.shape[0]*0.10)))
        parts.append(value.topk(k,dim=0,largest=True,sorted=False).values.mean(0))
    if pooling == "mean_max_std_topk_segments":
        parts.extend(chunk.mean(0) for chunk in torch.chunk(value,4,dim=0))
    if pooling not in ("mean_max", "mean_max_std", "mean_max_std_topk",
                        "mean_max_std_topk_segments"):
        raise ValueError("Unknown pooling: %s" % pooling)
    return torch.cat(parts, dim=0)


class PairDataset(Dataset):
    def __init__(self, frame, embeddings):
        self.frame = frame.reset_index(drop=True); self.embeddings = embeddings
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        a, b = self.embeddings[row.protein_A], self.embeddings[row.protein_B]
        return a, b, torch.tensor(float(row.label), dtype=torch.float32), row.pair_id


class SymmetricESMPairNet(nn.Module):
    def __init__(self, protein_dim=2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(protein_dim * 2), nn.Linear(protein_dim * 2, 512),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(512, 128), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(128, 1),
        )
    def forward(self, a, b):
        x = torch.cat([(a - b).abs(), a * b], dim=-1)
        return self.net(x).squeeze(-1)


class SiameseGatedESMPairNet(nn.Module):
    """Shared protein encoder plus order-invariant gated pair fusion.

    The shared bottleneck substantially reduces parameters relative to a raw 5120-D
    pair MLP and is better regularised for the small protein-cold RBP400 split.
    """
    def __init__(self, protein_dim=2560, hidden_dim=384, latent_dim=192):
        super().__init__()
        self.protein_encoder = nn.Sequential(
            nn.LayerNorm(protein_dim),
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        pair_dim = latent_dim * 3
        self.gate = nn.Sequential(nn.Linear(pair_dim, latent_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim + latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, a, b):
        za, zb = self.protein_encoder(a), self.protein_encoder(b)
        pair = torch.cat([(za - zb).abs(), za * zb, 0.5 * (za + zb)], dim=-1)
        gated = self.gate(pair) * (za * zb)
        return self.classifier(torch.cat([pair, gated], dim=-1)).squeeze(-1)


class SiameseCompactESMPairNet(nn.Module):
    """Low-rank symmetric encoder designed to reduce V2 overfitting."""
    def __init__(self, protein_dim=3840, latent_dim=128):
        super().__init__()
        self.protein_encoder = nn.Sequential(
            nn.LayerNorm(protein_dim),
            nn.Dropout(0.12),
            nn.Linear(protein_dim, latent_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.35),
        )
        pair_dim = latent_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 96),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(96, 1),
        )

    def forward(self, a, b):
        za, zb = self.protein_encoder(a), self.protein_encoder(b)
        pair = torch.cat([(za-zb).abs(), za*zb, 0.5*(za+zb)], dim=-1)
        return self.classifier(pair).squeeze(-1)


def build_model(architecture, protein_dim=2560):
    if architecture == "siamese_compact":
        return SiameseCompactESMPairNet(protein_dim=protein_dim)
    if architecture == "siamese_gated":
        return SiameseGatedESMPairNet(protein_dim=protein_dim)
    if architecture == "pair_mlp":
        return SymmetricESMPairNet(protein_dim=protein_dim)
    raise ValueError("Unknown architecture: %s" % architecture)


def collect(model, loader, device):
    model.eval(); logits=[]; labels=[]; pair_ids=[]
    with torch.no_grad():
        for a, b, y, pid in loader:
            logits.append(model(a.to(device), b.to(device)).cpu().numpy())
            labels.append(y.numpy()); pair_ids.extend(pid)
    return np.concatenate(logits), np.concatenate(labels).astype(int), pair_ids


def best_threshold(y, p):
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 199), p])
    best = None
    for threshold in candidates:
        pred = (p >= threshold).astype(int)
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else -1.0
        item = (mcc, f1_score(y, pred, zero_division=0), -abs(threshold - 0.5), threshold)
        if best is None or item > best: best = item
    return float(best[-1])


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
    }


def main():
    a = parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(a.device); a.output_root.mkdir(parents=True, exist_ok=True)
    names = ["train", "val"] + (["test"] if a.evaluate_test else [])
    frames = {name: pd.read_csv(a.split_root / f"{name}.tsv", sep="\t") for name in names}
    proteins = sorted(set().union(*[set(x.protein_A) | set(x.protein_B) for x in frames.values()]))
    embeddings = {p: load_embedding(a.esm_cache, p, a.pooling) for p in proteins}
    protein_dim = int(next(iter(embeddings.values())).numel())
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    records = []
    for seed in seeds:
        seed_all(seed); run = a.output_root / f"seed_{seed}"; run.mkdir(parents=True, exist_ok=True)
        datasets = {k: PairDataset(v, embeddings) for k, v in frames.items()}
        npos = int(frames["train"].label.sum()); nneg = len(frames["train"]) - npos
        if a.sampling == "balanced":
            labels = frames["train"].label.astype(int).to_numpy()
            sample_weights = np.where(labels == 1, 0.5 / max(1, npos), 0.5 / max(1, nneg))
            sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                            num_samples=len(labels), replacement=True)
            train_loader = DataLoader(datasets["train"], a.batch_size, sampler=sampler)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            train_loader = DataLoader(datasets["train"], a.batch_size, shuffle=True)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / max(1, npos), device=device))
        loaders = {"train": train_loader, "val": DataLoader(datasets["val"], a.batch_size, shuffle=False)}
        if "test" in datasets:
            loaders["test"] = DataLoader(datasets["test"], a.batch_size, shuffle=False)
        model = build_model(a.architecture, protein_dim=protein_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
        )
        best_ap=-1; stale=0; history=[]; start=time.time()
        for epoch in range(1, a.epochs + 1):
            model.train(); total=0.0
            for pa, pb, y, _ in loaders["train"]:
                target=y.to(device)
                if a.label_smoothing > 0:
                    target=target*(1.0-a.label_smoothing)+0.5*a.label_smoothing
                optimizer.zero_grad(); loss=loss_fn(model(pa.to(device), pb.to(device)), target); loss.backward()
                if a.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(y)
            val_logits, val_y, _ = collect(model, loaders["val"], device)
            val_ap = average_precision_score(val_y, val_logits)
            scheduler.step(val_ap)
            history.append({"epoch": epoch, "train_loss": total/len(datasets["train"]),
                            "val_raw_auprc": float(val_ap),
                            "learning_rate": float(optimizer.param_groups[0]["lr"])})
            if val_ap > best_ap + 1e-6:
                best_ap=val_ap; stale=0; torch.save(model.state_dict(), run / "best_state.pt")
            else:
                stale += 1
                if stale >= a.patience: break
        model.load_state_dict(torch.load(run / "best_state.pt", map_location=device))
        val_logits, val_y, val_ids = collect(model, loaders["val"], device)
        calibrator = LogisticRegression(solver="lbfgs").fit(val_logits.reshape(-1,1), val_y)
        val_prob = calibrator.predict_proba(val_logits.reshape(-1,1))[:,1]
        threshold = best_threshold(val_y, val_prob)
        evaluation_split = "test" if a.evaluate_test else "val"
        eval_logits, eval_y, eval_ids = collect(model, loaders[evaluation_split], device)
        eval_prob = calibrator.predict_proba(eval_logits.reshape(-1,1))[:,1]
        result = metrics(eval_y, eval_prob, threshold)
        result.update({"seed":seed, "evaluation_split":evaluation_split,
                       "training_seconds":time.time()-start, "parameters":sum(p.numel() for p in model.parameters())})
        torch.save({
            "state_dict": model.state_dict(), "seed": seed,
            "calibration_coef": float(calibrator.coef_[0,0]),
            "calibration_intercept": float(calibrator.intercept_[0]),
            "decision_threshold": threshold, "protein_embedding_dim": protein_dim,
            "model_class": type(model).__name__, "architecture": a.architecture,
            "sampling": a.sampling, "pooling": a.pooling,
            "label_smoothing": a.label_smoothing,
        }, run / "case_model.pt")
        pd.DataFrame(history).to_csv(run / "history.tsv", sep="\t", index=False)
        pd.DataFrame({"pair_id":eval_ids,"label":eval_y,"raw_logit":eval_logits,
                      "calibrated_probability":eval_prob}).to_csv(
                          run/(evaluation_split+"_predictions.tsv"),sep="\t",index=False)
        (run/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        records.append(result); print(json.dumps(result), flush=True)
    table=pd.DataFrame(records); table.to_csv(a.output_root/"metrics_by_seed.tsv",sep="\t",index=False)
    ddof = 1 if len(table) > 1 else 0
    summary={c:{"mean":float(table[c].mean()),"sd":float(table[c].std(ddof=ddof))} for c in ("auprc","auroc","mcc","f1","brier")}
    (a.output_root/"metrics_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
