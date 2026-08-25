# -*- coding: utf-8 -*-
"""Shared utilities for reproducible TUnA evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import current_data_config
from model import TRIAGEPPIModel
from train import (
    TriagePairDataset,
    batch_to_device,
    collate_pad,
    forward_batch,
    load_resume,
    split_dataset_train_val_test,
)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))


def outcome_name(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def gate_entropy(g_res: float, g_interface: float, g_pair: float) -> float:
    out = 0.0
    for g in (g_res, g_interface, g_pair):
        if g > 0:
            out -= g * math.log(g)
    return out


def tensor_list(x: torch.Tensor) -> List[float]:
    return x.detach().cpu().float().view(-1).tolist()


def subset_indices(ds) -> List[int]:
    if isinstance(ds, Subset):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))


def pair_label(base: TriagePairDataset, idx: int) -> float:
    row = base.rows.iloc[int(idx)]
    pair_id = str(row[base.pair_col]) if base.pair_col else f"{row[base.a_col]}__{row[base.b_col]}"
    if getattr(base, "pair_labels", None):
        if pair_id not in base.pair_labels:
            raise KeyError(f"Pair label missing for {pair_id}")
        return float(base.pair_labels[pair_id])
    if base.y_col:
        return float(row[base.y_col])
    if getattr(base, "contact_pair_labels", None):
        return float(base.contact_pair_labels.get(pair_id, 0.0))
    return float(pair_id in base.positive_pair_ids)


def pair_id_at(base: TriagePairDataset, idx: int) -> str:
    row = base.rows.iloc[int(idx)]
    return str(row[base.pair_col]) if base.pair_col else f"{row[base.a_col]}__{row[base.b_col]}"


def split_label_counts(base: TriagePairDataset, indices: Iterable[int]) -> Tuple[int, int]:
    pos = 0
    neg = 0
    for idx in indices:
        if pair_label(base, int(idx)) >= 0.5:
            pos += 1
        else:
            neg += 1
    return pos, neg


def dataset_fingerprint(base: TriagePairDataset, pair_fourpack_dir: str) -> Dict:
    chunks: List[str] = []
    labels = []
    for i in range(len(base.rows)):
        row = base.rows.iloc[i]
        pair_id = pair_id_at(base, i)
        a = str(row[base.a_col])
        b = str(row[base.b_col])
        y = int(pair_label(base, i) >= 0.5)
        labels.append(y)
        chunks.append(f"{pair_id}\t{a}\t{b}\t{y}")
    digest = hashlib.md5("\n".join(chunks).encode("utf-8")).hexdigest()
    n_pos = int(sum(labels))
    n = len(labels)
    return {
        "pair_fourpack_dir": str(pair_fourpack_dir),
        "n_rows": n,
        "n_positive": n_pos,
        "n_negative": n - n_pos,
        "positive_rate": n_pos / max(1, n),
        "first5_pair_id": [pair_id_at(base, i) for i in range(min(5, n))],
        "md5": digest,
    }


def save_split_manifest(path: str, base: TriagePairDataset, split_map: Dict[str, object], seed: int, split_mode: str, pair_fourpack_dir: str) -> None:
    out = {
        "seed": int(seed),
        "split_mode": split_mode,
        "dataset_fingerprint": dataset_fingerprint(base, pair_fourpack_dir),
        "splits": {},
    }
    split_proteins: Dict[str, set] = {}
    for name, ds in split_map.items():
        idx = subset_indices(ds)
        pos, neg = split_label_counts(base, idx)
        proteins = set()
        for i in idx:
            row = base.rows.iloc[int(i)]
            proteins.add(str(row[base.a_col]))
            proteins.add(str(row[base.b_col]))
        split_proteins[name] = proteins
        out["splits"][name] = {
            "indices": idx,
            "pair_ids": [pair_id_at(base, i) for i in idx],
            "protein_ids": sorted(proteins),
            "n_proteins": len(proteins),
            "n": len(idx),
            "positive": pos,
            "negative": neg,
            "positive_rate": pos / max(1, len(idx)),
        }
    out["protein_overlap_audit"] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = split_proteins.get(left, set()) & split_proteins.get(right, set())
        out["protein_overlap_audit"][f"{left}_vs_{right}"] = {
            "n_overlap": len(common),
            "first20": sorted(common)[:20],
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_split_manifest(path: str, base: TriagePairDataset) -> Dict[str, Subset]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    current_fp = dataset_fingerprint(base, str(getattr(base.cfg, "pair_fourpack_dir", "")))
    expected_md5 = str(
        obj.get("dataset_fingerprint_md5", "")
        or obj.get("dataset_fingerprint", {}).get("md5", "")
    )
    if expected_md5 and expected_md5 != current_fp["md5"]:
        raise ValueError(
            f"split manifest fingerprint mismatch: manifest={expected_md5} current={current_fp['md5']}"
        )
    current = {pair_id_at(base, i): i for i in range(len(base.rows))}
    out: Dict[str, Subset] = {}
    used_pair_ids: List[str] = []
    for name in ("train", "val", "test"):
        if name not in obj.get("splits", {}):
            raise KeyError(f"split manifest missing split: {name}")
        pair_ids = obj["splits"][name].get("pair_ids", [])
        missing = [pid for pid in pair_ids if pid not in current]
        if missing:
            raise ValueError(f"split manifest has {len(missing)} pair_ids not present in current data; first={missing[:5]}")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"split manifest contains duplicate pair_ids in {name}")
        used_pair_ids.extend(pair_ids)
        out[name] = Subset(base, [current[pid] for pid in pair_ids])
    if len(used_pair_ids) != len(set(used_pair_ids)):
        raise ValueError("split manifest overlaps across train/val/test")
    if set(used_pair_ids) != set(current):
        raise ValueError(
            f"split manifest is not an exact dataset partition: covered={len(set(used_pair_ids))} dataset={len(current)}"
        )
    return out


def build_tuna_cfg(args):
    fourpack = Path(getattr(args, "pair_fourpack_dir", "") or "data/TUnA/Intra0/fourpack")
    if not fourpack.exists():
        fourpack = Path("data/TUnA/Intra0/fourpack")
    esm_dir = Path(getattr(args, "pair_esm_dir", "") or fourpack / "emb" / "esm2")
    cfg = current_data_config(
        pair_fourpack_dir=str(fourpack),
        pair_esm_dir=str(esm_dir),
        batch_size=int(getattr(args, "batch_size", 16) or 16),
        num_workers=int(getattr(args, "num_workers", 0) if getattr(args, "num_workers", 0) >= 0 else 0),
        use_pair_pssm=True,
        use_pair_dssp_ss=True,
        use_pair_dssp_rsa=True,
        use_esm=True,
    )
    cfg.require_complete_pair_labels = True
    return cfg


def build_splits(base: TriagePairDataset, cfg, seed: int, split_mode: str, split_file: str = "", save_split_file: str = "") -> Dict[str, object]:
    if split_file:
        return load_split_manifest(split_file, base)
    train_ds, val_ds, test_ds = split_dataset_train_val_test(base, cfg, seed + 29, split_mode)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    if save_split_file:
        save_split_manifest(save_split_file, base, split_map, seed, split_mode, cfg.pair_fourpack_dir)
    return split_map


def make_loader(ds, cfg) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=cfg.num_workers)


def infer_tuna_predictions(model: TRIAGEPPIModel, loader: DataLoader, device: torch.device, split_name: str, threshold_auto: float) -> pd.DataFrame:
    rows: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pair_ids = list(batch["pair_id"])
            protein_a = list(batch["protein_A"])
            protein_b = list(batch["protein_B"])
            batch = batch_to_device(batch, device)
            out = forward_batch(model, batch)
            labels = tensor_list(batch["y_pair"])
            p_triage = tensor_list(out["p_triage"])
            p_pair_raw = tensor_list(out["p_pair_raw"])
            p_res = tensor_list(torch.sigmoid(out["logit_res"]))
            p_interface = tensor_list(torch.sigmoid(out["logit_interface"]))
            logit_triage = tensor_list(out["logit_triage"])
            logit_pair_raw = tensor_list(out["logit_pair_raw"])
            logit_res = tensor_list(out["logit_res"])
            logit_interface = tensor_list(out["logit_interface"])
            g_res = tensor_list(out["fusion_weights"]["g_res"])
            g_interface = tensor_list(out["fusion_weights"]["g_interface"])
            g_pair = tensor_list(out["fusion_weights"]["g_pair"])
            reliability = tensor_list(out["evidence_reliability"])
            for i, pair_id in enumerate(pair_ids):
                label = int(labels[i] >= 0.5)
                pred_0p5 = int(p_triage[i] >= 0.5)
                pred_auto = int(p_triage[i] >= threshold_auto)
                gr, gi, gp = g_res[i], g_interface[i], g_pair[i]
                rows.append(
                    {
                        "pair_id": pair_id,
                        "protein_A": protein_a[i],
                        "protein_B": protein_b[i],
                        "label": label,
                        "split": split_name,
                        "p_triage": p_triage[i],
                        "p_pair_raw": p_pair_raw[i],
                        "p_res": p_res[i],
                        "p_interface": p_interface[i],
                        "logit_triage": logit_triage[i],
                        "logit_pair_raw": logit_pair_raw[i],
                        "logit_res": logit_res[i],
                        "logit_interface": logit_interface[i],
                        "g_res": gr,
                        "g_interface": gi,
                        "g_pair": gp,
                        "gate_entropy": gate_entropy(gr, gi, gp),
                        "gate_max": max(gr, gi, gp),
                        "evidence_reliability": reliability[i],
                        "pred_0p5": pred_0p5,
                        "pred_auto": pred_auto,
                        "threshold_auto": threshold_auto,
                        "outcome_0p5": outcome_name(label, pred_0p5),
                        "outcome_auto": outcome_name(label, pred_auto),
                    }
                )
    return pd.DataFrame(rows)


def auc_roc_score(score, label) -> float:
    s = torch.tensor(list(score), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    valid = torch.isfinite(s) & torch.isfinite(y)
    s, y = s[valid], y[valid]
    n_pos = float(y.sum())
    n_neg = float(y.numel() - y.sum())
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / max(1e-8, n_pos * n_neg))


def auprc_score(score, label) -> float:
    s = torch.tensor(list(score), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    valid = torch.isfinite(s) & torch.isfinite(y)
    s, y = s[valid], y[valid]
    if y.numel() == 0 or float(y.sum()) <= 0:
        return float("nan")
    order = torch.argsort(s, descending=True)
    yy = y[order]
    precision = torch.cumsum(yy, 0) / torch.arange(1, yy.numel() + 1, dtype=torch.float32)
    return float((precision * yy).sum() / yy.sum())


def metrics_at_threshold(score, label, threshold: float) -> Dict[str, float]:
    s = torch.tensor(list(score), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    pred = (s >= float(threshold)).float()
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    bal_acc = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    denom = math.sqrt(max(1e-8, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {
        "ACC": acc,
        "Balanced_ACC": bal_acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "MCC": mcc,
        "threshold_used": float(threshold),
        "pred_pos_rate": float(pred.mean()) if pred.numel() else float("nan"),
    }


def full_metrics(score, label, threshold: float) -> Dict[str, float]:
    out = {
        "AUC": auc_roc_score(score, label),
        "AUPRC": auprc_score(score, label),
    }
    out.update(metrics_at_threshold(score, label, threshold))
    return out


def best_threshold(score, label, objective: str = "mcc") -> Tuple[float, Dict[str, float]]:
    s = torch.tensor(list(score), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    valid = torch.isfinite(s) & torch.isfinite(y)
    s, y = s[valid], y[valid]
    if s.numel() == 0:
        return 0.5, full_metrics(score, label, 0.5)

    # Fast exact scan over score cut points. This replaces the old
    # candidate-by-candidate metric recomputation, which was very slow inside
    # static-weight grid search.
    order = torch.argsort(s, descending=True)
    ss = s[order]
    yy = y[order]
    pos_total = float(yy.sum())
    neg_total = float(yy.numel() - yy.sum())
    tp_cum = torch.cumsum(yy, 0)
    fp_cum = torch.cumsum(1.0 - yy, 0)
    group_end = torch.ones_like(ss, dtype=torch.bool)
    group_end[:-1] = ss[:-1] != ss[1:]
    ends = torch.nonzero(group_end, as_tuple=False).view(-1)

    tp = tp_cum[ends]
    fp = fp_cum[ends]
    fn = pos_total - tp
    tn = neg_total - fp
    recall = tp / torch.clamp(tp + fn, min=1.0)
    specificity = tn / torch.clamp(tn + fp, min=1.0)
    bal_acc = 0.5 * (recall + specificity)
    denom = torch.sqrt(torch.clamp((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), min=1e-8))
    mcc = (tp * tn - fp * fn) / denom
    key = mcc if objective == "mcc" else bal_acc
    best_i = int(torch.argmax(key).item())
    best_t = float(ss[ends[best_i]].item())

    m05 = full_metrics(s.tolist(), y.tolist(), 0.5)
    best_m = full_metrics(s.tolist(), y.tolist(), best_t)
    best_key = best_m["MCC"] if objective == "mcc" else best_m["Balanced_ACC"]
    key05 = m05["MCC"] if objective == "mcc" else m05["Balanced_ACC"]
    if key05 > best_key:
        return 0.5, m05
    return best_t, best_m


def brier_score(score, label) -> float:
    s = torch.tensor(list(score), dtype=torch.float32)
    y = torch.tensor(list(label), dtype=torch.float32)
    return float(((s - y) ** 2).mean())


def nll_score(score, label) -> float:
    s = torch.tensor(list(score), dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    y = torch.tensor(list(label), dtype=torch.float32)
    return float(-(y * s.log() + (1 - y) * (1 - s).log()).mean())


def ece_bins(score, label, n_bins: int = 10) -> Tuple[float, float, pd.DataFrame]:
    s = torch.tensor(list(score), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    rows = []
    ece = 0.0
    mce = 0.0
    total = max(1, int(s.numel()))
    for bi in range(n_bins):
        lo, hi = bi / n_bins, (bi + 1) / n_bins
        mask = ((s >= lo) & (s <= hi)) if bi == n_bins - 1 else ((s >= lo) & (s < hi))
        n = int(mask.sum())
        if n:
            mean_pred = float(s[mask].mean())
            obs = float(y[mask].mean())
            gap = abs(obs - mean_pred)
            ece += n / total * gap
            mce = max(mce, gap)
        else:
            mean_pred = obs = gap = float("nan")
        rows.append({"bin": bi, "bin_low": lo, "bin_high": hi, "n": n, "mean_pred": mean_pred, "observed_positive_rate": obs, "abs_gap": gap})
    return ece, mce, pd.DataFrame(rows)


def fit_temperature_from_val(logits, label) -> Tuple[float, bool]:
    x = torch.tensor(list(logits), dtype=torch.float32).view(-1)
    y = torch.tensor(list(label), dtype=torch.float32).view(-1)
    candidates = torch.logspace(math.log10(0.2), math.log10(100.0), steps=240)
    best_t = 1.0
    best_nll = float("inf")
    for t in candidates:
        p = torch.sigmoid(x / t).clamp(1e-7, 1 - 1e-7)
        nll = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    at_boundary = best_t <= 0.201 or best_t >= 99.0
    return best_t, at_boundary


def sigmoid_series(x) -> pd.Series:
    return pd.Series(torch.sigmoid(torch.tensor(list(x), dtype=torch.float32)).numpy())


def add_basic_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mean_prob_fusion"] = (out["p_res"] + out["p_interface"] + out["p_pair_raw"]) / 3.0
    out["mean_logit_fusion"] = sigmoid_series((out["logit_res"] + out["logit_interface"] + out["logit_pair_raw"]) / 3.0)
    return out


def fit_static_weights(val_df: pd.DataFrame, objective: str = "mcc") -> Dict:
    y = val_df["label"].astype(float)
    best = None
    grid = [i / 20.0 for i in range(21)]
    tried = 0
    for w_res in grid:
        for w_interface in grid:
            w_pair = 1.0 - w_res - w_interface
            if w_pair < -1e-9:
                continue
            tried += 1
            if tried == 1 or tried % 50 == 0:
                print(f"[static-weight] grid {tried}/231", flush=True)
            logit = w_res * val_df["logit_res"] + w_interface * val_df["logit_interface"] + w_pair * val_df["logit_pair_raw"]
            score = sigmoid_series(logit)
            thr, m = best_threshold(score, y, objective=objective)
            key = m["MCC"] if objective == "mcc" else m["Balanced_ACC"]
            if best is None or key > best["select"]:
                best = {"w_res": w_res, "w_interface": w_interface, "w_pair": w_pair, "threshold": thr, "select": key}
    return best or {"w_res": 1 / 3, "w_interface": 1 / 3, "w_pair": 1 / 3, "threshold": 0.5, "select": 0.0}


def apply_static_weights(df: pd.DataFrame, weights: Dict) -> pd.Series:
    logit = weights["w_res"] * df["logit_res"] + weights["w_interface"] * df["logit_interface"] + weights["w_pair"] * df["logit_pair_raw"]
    return sigmoid_series(logit)


def fit_logistic_stacking(val_df: pd.DataFrame, max_iter: int = 500) -> Dict:
    x = torch.tensor(val_df[["logit_res", "logit_interface", "logit_pair_raw"]].astype(float).to_numpy(), dtype=torch.float32)
    y = torch.tensor(val_df["label"].astype(float).to_numpy(), dtype=torch.float32).view(-1, 1)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    xs = (x - mean) / std
    w = torch.zeros((3, 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xs @ w + b, y)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        score = torch.sigmoid(xs @ w + b).view(-1).numpy()
    thr, _ = best_threshold(score, val_df["label"].astype(float), objective="mcc")
    return {
        "coef_res": float(w[0, 0]),
        "coef_interface": float(w[1, 0]),
        "coef_pair": float(w[2, 0]),
        "intercept": float(b[0]),
        "mean_res": float(mean[0, 0]),
        "mean_interface": float(mean[0, 1]),
        "mean_pair": float(mean[0, 2]),
        "std_res": float(std[0, 0]),
        "std_interface": float(std[0, 1]),
        "std_pair": float(std[0, 2]),
        "threshold": float(thr),
    }


def apply_logistic_stacking(df: pd.DataFrame, params: Dict) -> pd.Series:
    x = torch.tensor(df[["logit_res", "logit_interface", "logit_pair_raw"]].astype(float).to_numpy(), dtype=torch.float32)
    mean = torch.tensor([[params["mean_res"], params["mean_interface"], params["mean_pair"]]], dtype=torch.float32)
    std = torch.tensor([[params["std_res"], params["std_interface"], params["std_pair"]]], dtype=torch.float32).clamp_min(1e-6)
    w = torch.tensor([[params["coef_res"]], [params["coef_interface"]], [params["coef_pair"]]], dtype=torch.float32)
    b = torch.tensor([params["intercept"]], dtype=torch.float32)
    return pd.Series(torch.sigmoid(((x - mean) / std) @ w + b).view(-1).numpy())


def write_tsv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def read_pred(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    for col in df.columns:
        if col not in ("pair_id", "protein_A", "protein_B", "split", "outcome_0p5", "outcome_auto"):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def load_model_for_tuna(cfg, checkpoint: str, device: torch.device) -> TRIAGEPPIModel:
    model = TRIAGEPPIModel(cfg).to(device)
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    load_resume(model, checkpoint, device)
    return model
