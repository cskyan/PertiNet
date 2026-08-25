# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import current_data_config
try:
    from config import gpsite_data_config
except ImportError:
    gpsite_data_config = None
try:
    from config import dest_data_config
except ImportError:
    dest_data_config = None
try:
    from config import rbp400_data_config
except ImportError:
    rbp400_data_config = None
from model import TRIAGEPPIModel
from train import (
    GraphRBFPPDataset,
    STAGE_CKPT,
    TriagePairDataset,
    batch_to_device,
    binary_metrics,
    binary_metrics_best_threshold,
    collate_pad,
    apply_l1_ager,
    l1_label_bucket,
    residue_topk_metrics,
    split_dataset_train_val_test,
    forward_batch,
    load_resume,
)
from tuna_utils import (
    best_threshold as tuna_best_threshold,
    build_splits as tuna_build_splits,
    dataset_fingerprint as tuna_dataset_fingerprint,
    full_metrics as tuna_full_metrics,
    infer_tuna_predictions,
    make_loader as tuna_make_loader,
    pair_label as tuna_pair_label,
    save_split_manifest as tuna_save_split_manifest,
    split_label_counts as tuna_split_label_counts,
    write_tsv as tuna_write_tsv,
)


CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
DEFAULT_PP_CKPT = f"{CHECKPOINT_DIR}/triage_l1_graphrbf_auc.pt"
DEFAULT_GPSITE_CKPT = f"{CHECKPOINT_DIR}/triage_l1_gpsite_auc.pt"
DEFAULT_DEST_CKPT = f"{CHECKPOINT_DIR}/triage_l1_dest.pt"
DEFAULT_RBP400_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary.pt"
DEFAULT_RBP400_SCORE_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_score.pt"
DEFAULT_RBP400_ACC_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_acc.pt"
DEFAULT_RBP400_PRECISION_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_precision.pt"
DEFAULT_RBP400_RECALL_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_recall.pt"
DEFAULT_RBP400_F1_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_f1.pt"
DEFAULT_RBP400_AUROC_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_auroc.pt"
DEFAULT_RBP400_AUPRC_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_auprc.pt"
DEFAULT_RBP400_MCC_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_binary_best_mcc.pt"
DEFAULT_RBP400_SWA_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_pareto_swa.pt"
DEFAULT_TOPK_ENGINEERING_CKPT = f"{CHECKPOINT_DIR}/triage_topk_final.pt"
DEFAULT_RBP400_TRIAGE_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage.pt"
DEFAULT_RBP400_TRIAGE_SCORE_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_score.pt"
DEFAULT_RBP400_TRIAGE_TARGET_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_target.pt"
DEFAULT_RBP400_TRIAGE_RECALL_L5_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_recall_l5.pt"
DEFAULT_RBP400_TRIAGE_RECALL_L10_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_recall_l10.pt"
DEFAULT_RBP400_TRIAGE_PRECISION_10_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_precision_10.pt"
DEFAULT_RBP400_TRIAGE_P10_R10_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_best_p10_r10.pt"
DEFAULT_RBP400_TRIAGE_SWA_CKPT = f"{CHECKPOINT_DIR}/triage_rbp400_triage_pareto_swa.pt"
DEFAULT_RBP400_TOPK_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk.pt"
DEFAULT_RBP400_TOPK_SCORE_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_score.pt"
DEFAULT_RBP400_TOPK_TARGET_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_target.pt"
DEFAULT_RBP400_TOPK_RECALL_L5_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_recall_l5.pt"
DEFAULT_RBP400_TOPK_RECALL_L10_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_recall_l10.pt"
DEFAULT_RBP400_TOPK_PRECISION_10_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_precision_10.pt"
DEFAULT_RBP400_TOPK_P10_R10_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_best_p10_r10.pt"
DEFAULT_RBP400_TOPK_SWA_CKPT = f"{CHECKPOINT_DIR}/triage_l1_rbp400_topk_pareto_swa.pt"
DEFAULT_TUNA_PAIR_CKPT = f"{CHECKPOINT_DIR}/triage_tuna_pair_finetuned.pt"
DEFAULT_PP_PRED_TSV = f"{OUTPUT_DIR}/l1_test_predictions.tsv"
DEFAULT_GPSITE_PRED_TSV = f"{OUTPUT_DIR}/gpsite_test_predictions.tsv"
DEFAULT_DEST_PRED_TSV = f"{OUTPUT_DIR}/dset_test_predictions.tsv"
DEFAULT_RBP400_PRED_TSV = f"{OUTPUT_DIR}/rbp400_test_predictions.tsv"
DEFAULT_TUNA_PAIR_PRED_TSV = f"{OUTPUT_DIR}/tuna_pair_test_predictions.tsv"


def rbp400_preset_checkpoints(preset: str) -> str:
    mapping = {
        "paper": DEFAULT_RBP400_PRECISION_CKPT,
        "best": DEFAULT_RBP400_PRECISION_CKPT,
        "score": DEFAULT_RBP400_SCORE_CKPT,
        "acc": DEFAULT_RBP400_ACC_CKPT,
        "precision": DEFAULT_RBP400_PRECISION_CKPT,
        "recall": DEFAULT_RBP400_RECALL_CKPT,
        "f1": DEFAULT_RBP400_F1_CKPT,
        "auroc": DEFAULT_RBP400_AUROC_CKPT,
        "auprc": DEFAULT_RBP400_AUPRC_CKPT,
        "mcc": DEFAULT_RBP400_MCC_CKPT,
        "swa": DEFAULT_RBP400_SWA_CKPT,
        "ensemble_best_swa": f"{DEFAULT_RBP400_CKPT},{DEFAULT_RBP400_SWA_CKPT}",
        "ensemble_best_mcc": f"{DEFAULT_RBP400_CKPT},{DEFAULT_RBP400_MCC_CKPT}",
        "ensemble_best_precision": f"{DEFAULT_RBP400_CKPT},{DEFAULT_RBP400_PRECISION_CKPT}",
        "topk": DEFAULT_RBP400_TOPK_CKPT,
        "topk_engineering": DEFAULT_TOPK_ENGINEERING_CKPT,
        "triage": DEFAULT_RBP400_TRIAGE_CKPT,
        "triage_score": DEFAULT_RBP400_TRIAGE_SCORE_CKPT,
        "triage_target": DEFAULT_RBP400_TRIAGE_TARGET_CKPT,
        "triage_recall_l5": DEFAULT_RBP400_TRIAGE_RECALL_L5_CKPT,
        "triage_recall_l10": DEFAULT_RBP400_TRIAGE_RECALL_L10_CKPT,
        "triage_precision_10": DEFAULT_RBP400_TRIAGE_PRECISION_10_CKPT,
        "triage_p10_r10": DEFAULT_RBP400_TRIAGE_P10_R10_CKPT,
        "triage_swa": DEFAULT_RBP400_TRIAGE_SWA_CKPT,
        "triage_ensemble_best_swa": f"{DEFAULT_RBP400_TRIAGE_CKPT},{DEFAULT_RBP400_TRIAGE_SWA_CKPT}",
        "topk_score": DEFAULT_RBP400_TOPK_SCORE_CKPT,
        "topk_target": DEFAULT_RBP400_TOPK_TARGET_CKPT,
        "topk_recall_l5": DEFAULT_RBP400_TOPK_RECALL_L5_CKPT,
        "topk_recall_l10": DEFAULT_RBP400_TOPK_RECALL_L10_CKPT,
        "topk_precision_10": DEFAULT_RBP400_TOPK_PRECISION_10_CKPT,
        "topk_p10_r10": DEFAULT_RBP400_TOPK_P10_R10_CKPT,
        "topk_swa": DEFAULT_RBP400_TOPK_SWA_CKPT,
        "topk_ensemble_best_swa": f"{DEFAULT_RBP400_TOPK_CKPT},{DEFAULT_RBP400_TOPK_SWA_CKPT}",
    }
    return mapping[preset]


def parse_csv_values(value: str, cast=str) -> List:
    out = []
    for chunk in (value or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        out.append(cast(item))
    return out


def split_checkpoint_list(value: str) -> List[str]:
    out: List[str] = []
    for chunk in (value or "").replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def rbp400_weighted_score(metrics: Dict[str, float]) -> float:
    """Return the RBP400 checkpoint-selection score used during training."""
    acc = float(metrics.get("acc", metrics.get("l1_acc", 0.0)))
    auroc = float(metrics.get("auroc", metrics.get("l1_auroc", 0.0)))
    auprc = float(metrics.get("auprc", metrics.get("l1_auprc", 0.0)))
    mcc = max(0.0, float(metrics.get("mcc", metrics.get("l1_mcc", 0.0))))
    f1 = float(metrics.get("f1", metrics.get("l1_f1", 0.0)))
    return float(0.45 * acc + 0.25 * mcc + 0.15 * auroc + 0.10 * auprc + 0.05 * f1)


def rbp400_target_score(metrics: Dict[str, float]) -> float:
    return rbp400_weighted_score(metrics)


def base_config(dataset: str):
    """Use the correct model-facing config for each L1 dataset.

    For GPSite, this must be gpsite_data_config() rather than current_data_config(),
    because the balanced v6 checkpoint/model uses the GPSite-specific refiner and
    threshold policy. Checkpoint config is still preferred when available.
    """
    if dataset == "gpsite":
        if gpsite_data_config is None:
            raise ImportError("config.py does not provide gpsite_data_config(); use a checkpoint/config that includes GPSite support.")
        return gpsite_data_config()
    if dataset == "dest":
        if dest_data_config is None:
            raise ImportError("config.py does not provide dest_data_config(); use a checkpoint/config that includes Dest support.")
        return dest_data_config()
    if dataset == "rbp400":
        if rbp400_data_config is None:
            raise ImportError("config.py does not provide rbp400_data_config(); use a checkpoint/config that includes RBP400 support.")
        return rbp400_data_config()
    return current_data_config()


def config_from_checkpoint(checkpoint: str, dataset: str, use_current_config: bool = False):
    cfg = base_config(dataset)
    if use_current_config:
        return cfg, "current"
    ckpt = torch.load(checkpoint, map_location="cpu")
    saved_cfg = ckpt.get("config", {})
    if isinstance(saved_cfg, dict):
        loaded = 0
        for key, value in saved_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                loaded += 1
        return cfg, f"checkpoint({loaded})"
    return cfg, "current"


def metric_objective_weights(cfg) -> Dict[str, float]:
    """Match the threshold-selection objective used during GPSite L1 training."""
    return {
        "acc": float(getattr(cfg, "l1_score_w_acc", 0.0)),
        "precision": float(getattr(cfg, "l1_score_w_precision", 0.0)),
        "recall": float(getattr(cfg, "l1_score_w_recall", 0.0)),
        "f1": float(getattr(cfg, "l1_score_w_f1", 0.0)),
        "mcc": float(getattr(cfg, "l1_score_w_mcc", 0.0)),
    }


def smooth_by_sequence(prob: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return prob
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x = prob.float().unsqueeze(1)
    m = (mask > 0.5).float().unsqueeze(1)
    kernel = torch.ones(1, 1, window, device=prob.device, dtype=prob.dtype)
    num = F.conv1d(x * m, kernel, padding=pad).squeeze(1)
    den = F.conv1d(m, kernel, padding=pad).squeeze(1).clamp_min(1.0)
    return torch.where(mask > 0.5, num / den, prob)


def peak_boost_by_sequence(prob: torch.Tensor, mask: torch.Tensor, weight: float, window: int = 11) -> torch.Tensor:
    if weight <= 0:
        return prob
    base = smooth_by_sequence(prob, mask, window)
    boosted = prob + float(weight) * (prob - base).clamp_min(0.0)
    return torch.where(mask > 0.5, boosted.clamp(0.0, 1.0), prob)


@torch.no_grad()
def collect_l1_predictions(
    model,
    loader,
    device,
    smooth_window: int = 1,
    cfg=None,
    use_ager: bool = False,
    peak_boost: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    models = model if isinstance(model, list) else [model]
    for m in models:
        m.eval()
    probs, labels = [], []
    rows: List[Dict] = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        model_probs = []
        for m in models:
            out = forward_batch(m, batch)
            model_probs.append(out["p_res_A"])
        raw_prob = torch.stack(model_probs, dim=0).mean(dim=0)
        prob = raw_prob
        if use_ager and cfg is not None:
            prob = apply_l1_ager(prob, batch.get("coordsA"), batch["maskA"] > 0.5, cfg)
        prob = smooth_by_sequence(prob, batch["maskA"], smooth_window)
        prob = peak_boost_by_sequence(prob, batch["maskA"], peak_boost)
        label = batch["y_res_A"]
        mask = batch["maskA"] > 0.5
        probs.append(prob.detach().cpu()[mask.detach().cpu()])
        labels.append(label.detach().cpu()[mask.detach().cpu()])

        raw_cpu = raw_prob.detach().cpu()
        prob_cpu = prob.detach().cpu()
        label_cpu = label.detach().cpu()
        mask_cpu = mask.detach().cpu()
        pids = batch.get("protein_A", [""] * prob_cpu.size(0))
        for b, pid in enumerate(pids):
            valid_len = int(mask_cpu[b].sum().item())
            for i in range(valid_len):
                rows.append({
                    "pid": pid,
                    "res_index": i + 1,
                    "label": int(label_cpu[b, i].item() > 0.5),
                    "prob": float(prob_cpu[b, i].item()),
                    "raw_prob": float(raw_cpu[b, i].item()),
                })
    if not probs:
        return torch.empty(0), torch.empty(0), rows
    return torch.cat(probs), torch.cat(labels), rows


def make_loader(cfg, split: str, batch_size: int, num_workers: int, dataset: str):
    if dataset == "gpsite":
        ds = GraphRBFPPDataset(cfg, cfg.gpsite_root, split, max_items=0, esm_dir=cfg.gpsite_esm_dir, dataset_label="GPSite-PRO")
    elif dataset == "dest":
        ds = GraphRBFPPDataset(cfg, cfg.dest_root, split, max_items=0, esm_dir=cfg.dest_esm_dir, dataset_label="Dest")
    elif dataset == "rbp400":
        ds = GraphRBFPPDataset(cfg, cfg.rbp400_root, split, max_items=0, esm_dir=cfg.rbp400_esm_dir, dataset_label="RBP400")
    else:
        ds = GraphRBFPPDataset(cfg, cfg.pp_root, split, max_items=0, esm_dir=cfg.pp_esm_dir, dataset_label="GraphRBF-PP")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=num_workers)


def write_rows(path: str, rows: List[Dict], threshold: float):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pid", "res_index", "label", "prob", "raw_prob", "pred"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["pred"] = int(float(row["prob"]) >= threshold)
            w.writerow(out)


def write_rank_rows(path: str, rows: List[Dict], threshold: Optional[float] = None):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[str(row["pid"])].append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pid",
            "res_index",
            "label",
            "prob",
            "raw_prob",
            "rank",
            "in_top2",
            "in_top10",
            "in_top20",
            "in_top_l5",
            "in_top_l10",
        ]
        if threshold is not None:
            fieldnames += ["pred", "threshold"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for pid in sorted(by_pid):
            vals = sorted(by_pid[pid], key=lambda r: float(r["prob"]), reverse=True)
            L = len(vals)
            top_l5 = max(1, int((L + 4) // 5))
            top_l10 = max(1, int((L + 9) // 10))
            for rank, row in enumerate(vals, start=1):
                out = {
                    "pid": row["pid"],
                    "res_index": row["res_index"],
                    "label": row["label"],
                    "prob": row["prob"],
                    "raw_prob": row.get("raw_prob", row["prob"]),
                    "rank": rank,
                    "in_top2": int(rank <= 2),
                    "in_top10": int(rank <= min(10, L)),
                    "in_top20": int(rank <= min(20, L)),
                    "in_top_l5": int(rank <= top_l5),
                    "in_top_l10": int(rank <= top_l10),
                }
                if threshold is not None:
                    out["pred"] = int(float(row["prob"]) >= float(threshold))
                    out["threshold"] = float(threshold)
                w.writerow(out)


def print_metrics(tag: str, metrics: Dict[str, float], threshold: float):
    print(
        f"[{tag}] "
        f"ACC={metrics['acc']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"AUROC={metrics['auroc']:.4f} "
        f"AUPRC={metrics['auprc']:.4f} "
        f"MCC={metrics['mcc']:.4f} "
        f"thr={threshold:.4f}",
        flush=True,
    )


def topk_metrics_from_rows(rows: List[Dict]) -> Dict[str, float]:
    by_pid = defaultdict(list)
    for row in rows:
        by_pid[str(row["pid"])].append(row)
    sums = {"l1_recall_l5": 0.0, "l1_recall_l10": 0.0, "l1_precision_10": 0.0, "l1_hit_2": 0.0, "l1_hit_20": 0.0}
    n = 0
    for vals in by_pid.values():
        vals = sorted(vals, key=lambda r: float(r["prob"]), reverse=True)
        labels = [1 if int(v["label"]) > 0 else 0 for v in vals]
        positives = sum(labels)
        if positives <= 0:
            continue
        L = len(labels)

        def hits(k: int) -> int:
            kk = max(1, min(int(k), L))
            return sum(labels[:kk])

        k_l5 = max(1, int((L + 4) // 5))
        k_l10 = max(1, int((L + 9) // 10))
        sums["l1_recall_l5"] += hits(k_l5) / max(1, positives)
        sums["l1_recall_l10"] += hits(k_l10) / max(1, positives)
        sums["l1_precision_10"] += hits(10) / max(1, min(10, L))
        sums["l1_hit_2"] += 1.0 if hits(2) > 0 else 0.0
        sums["l1_hit_20"] += 1.0 if hits(20) > 0 else 0.0
        n += 1
    if n <= 0:
        return {**sums, "l1_topk_n": 0.0}
    out = {k: v / n for k, v in sums.items()}
    out["l1_topk_n"] = float(n)
    return out


def print_topk_metrics(tag: str, metrics: Dict[str, float]):
    print(
        f"[{tag}/topk] "
        f"R@L/5={metrics['l1_recall_l5']:.4f} "
        f"R@L/10={metrics['l1_recall_l10']:.4f} "
        f"P@10={metrics['l1_precision_10']:.4f} "
        f"Hit@20={metrics.get('l1_hit_20', 0.0):.4f} "
        f"Hit@2={metrics['l1_hit_2']:.4f} "
        f"proteins={int(metrics['l1_topk_n'])}",
        flush=True,
    )


def write_l1_summary(path: str, rows: List[Dict]) -> None:
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def print_rbp400_bucket_metrics(cfg, tag: str, rows: List[Dict], threshold: float) -> None:
    root = Path(getattr(cfg, "rbp400_root", ""))
    if not root:
        return
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("pid", ""))
        bucket = l1_label_bucket(root, pid) if pid else "unknown"
        grouped[bucket].append(row)
    for bucket in sorted(grouped):
        vals = grouped[bucket]
        if not vals:
            continue
        prob = torch.tensor([float(v["prob"]) for v in vals], dtype=torch.float32)
        lab = torch.tensor([float(v["label"]) for v in vals], dtype=torch.float32)
        proteins = len({str(v.get("pid", "")) for v in vals})
        pos_rate = float(lab.mean().item()) if lab.numel() else 0.0
        metrics = binary_metrics(prob, lab, thr=threshold)
        print(
            f"[{tag}/bucket] bucket={bucket} proteins={proteins} residues={lab.numel()} "
            f"pos_rate={pos_rate:.4f} ACC={metrics['acc']:.4f} "
            f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} AUROC={metrics['auroc']:.4f} "
            f"AUPRC={metrics['auprc']:.4f} MCC={metrics['mcc']:.4f}",
            flush=True,
        )


def rbp400_sweep_sort_key(metrics: Dict[str, float], selection: str) -> float:
    if selection == "acc":
        return float(metrics.get("acc", 0.0))
    if selection == "precision":
        return float(metrics.get("precision", 0.0))
    if selection == "recall":
        return float(metrics.get("recall", 0.0))
    if selection == "f1":
        return float(metrics.get("f1", 0.0))
    if selection == "auroc":
        return float(metrics.get("auroc", 0.0))
    if selection == "auprc":
        return float(metrics.get("auprc", 0.0))
    if selection == "mcc":
        return float(metrics.get("mcc", 0.0))
    return rbp400_weighted_score(metrics)


def load_model_group(cfg, checkpoint_value: str, device):
    models = []
    loaded_paths = []
    paths = split_checkpoint_list(checkpoint_value)
    if not paths:
        raise ValueError("empty checkpoint group")
    for ckpt_path in paths:
        model = TRIAGEPPIModel(cfg).to(device)
        missing, _, skipped = load_resume(model, ckpt_path, device)
        if missing or skipped:
            print(
                f"[checkpoint-skip] incompatible checkpoint missing={len(missing)} "
                f"skipped_shape={len(skipped)} path={ckpt_path}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        models.append(model)
        loaded_paths.append(ckpt_path)
    if not models:
        raise RuntimeError(f"no compatible checkpoints loaded from: {checkpoint_value}")
    return models if len(models) > 1 else models[0], loaded_paths


def subset_indices(ds) -> List[int]:
    if isinstance(ds, Subset):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))


def subset_uids(base: TriagePairDataset, ds) -> set:
    out = set()
    for idx in subset_indices(ds):
        r = base.rows.iloc[int(idx)]
        out.add(str(r[base.a_col]))
        out.add(str(r[base.b_col]))
    return out


def pair_label_rate(base: TriagePairDataset, ds) -> Tuple[int, int, float]:
    idx = subset_indices(ds)
    pos, neg = tuna_split_label_counts(base, idx)
    return pos, neg, pos / max(1, len(idx))


@torch.no_grad()
def predict_tuna_pair(args, cfg, cfg_source: str) -> None:
    current_cfg = current_data_config()
    cfg.pair_fourpack_dir = current_cfg.pair_fourpack_dir
    cfg.pair_esm_dir = current_cfg.pair_esm_dir
    cfg.use_pair_pssm = current_cfg.use_pair_pssm
    cfg.use_pair_dssp_ss = current_cfg.use_pair_dssp_ss
    cfg.use_pair_dssp_rsa = current_cfg.use_pair_dssp_rsa
    cfg.batch_size = args.batch_size if args.batch_size > 0 else int(getattr(current_cfg, "batch_size", 16))
    cfg.num_workers = args.num_workers if args.num_workers >= 0 else int(getattr(current_cfg, "num_workers", 8))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    base = TriagePairDataset(cfg)
    splits = tuna_build_splits(base, cfg, args.seed, args.split_mode, split_file=args.split_file, save_split_file=args.save_split_file)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    eval_ds = splits[args.eval_split]

    train_uids = subset_uids(base, train_ds)
    val_uids = subset_uids(base, val_ds)
    test_uids = subset_uids(base, test_ds)
    print(
        f"[config] source={cfg_source} dataset=tuna_pair checkpoint={args.checkpoint} "
        f"split_mode={args.split_mode} calibrate={args.calibrate_split} eval={args.eval_split} "
        f"batch_size={cfg.batch_size} workers={cfg.num_workers}",
        flush=True,
    )
    print(f"[data-root] pair_fourpack_dir={cfg.pair_fourpack_dir}", flush=True)
    print(f"[data-root] pair_esm_dir={cfg.pair_esm_dir}", flush=True)
    print(f"[fingerprint] {tuna_dataset_fingerprint(base, cfg.pair_fourpack_dir)}", flush=True)
    if args.split_file:
        print(f"[split-file] loaded={args.split_file}", flush=True)
    if args.save_split_file:
        print(f"[split-file] saved={args.save_split_file}", flush=True)
    print(f"[split] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    for name, ds in splits.items():
        pos, neg, pr = pair_label_rate(base, ds)
        print(f"[labels] {name} pos={pos} neg={neg} pos_rate={pr:.4f}", flush=True)
    print(f"[leak-check] train_val_uid_overlap={len(train_uids & val_uids)}", flush=True)
    print(f"[leak-check] train_test_uid_overlap={len(train_uids & test_uids)}", flush=True)
    print(f"[leak-check] val_test_uid_overlap={len(val_uids & test_uids)}", flush=True)
    if args.report_only:
        return

    model = TRIAGEPPIModel(cfg).to(device)
    load_resume(model, args.checkpoint, device)
    model.eval()

    cal_df = infer_tuna_predictions(model, tuna_make_loader(splits[args.calibrate_split], cfg), device, args.calibrate_split, 0.5)
    if args.threshold == "auto":
        threshold, cal_metrics = tuna_best_threshold(cal_df["p_triage"], cal_df["label"], objective="mcc")
        print(f"[threshold-auto] fit_split={args.calibrate_split} threshold={threshold:.6f} val_MCC={cal_metrics['MCC']:.4f} val_BalACC={cal_metrics['Balanced_ACC']:.4f}", flush=True)
    else:
        threshold = float(args.threshold)
    eval_df = infer_tuna_predictions(model, tuna_make_loader(eval_ds, cfg), device, args.eval_split, threshold)
    metrics = tuna_full_metrics(eval_df["p_triage"], eval_df["label"], threshold)
    metrics_05 = tuna_full_metrics(eval_df["p_triage"], eval_df["label"], 0.5)
    print(
        f"[metrics] eval/{args.eval_split} score=p_triage thr={threshold:.6f} "
        f"AUC={metrics['AUC']:.4f} AUPRC={metrics['AUPRC']:.4f} "
        f"BalACC={metrics['Balanced_ACC']:.4f} F1={metrics['F1']:.4f} MCC={metrics['MCC']:.4f}",
        flush=True,
    )
    print(
        f"[metrics-0p5] eval/{args.eval_split} score=p_triage thr=0.500000 "
        f"BalACC={metrics_05['Balanced_ACC']:.4f} F1={metrics_05['F1']:.4f} MCC={metrics_05['MCC']:.4f}",
        flush=True,
    )
    p = torch.tensor(eval_df["p_triage"].astype(float).to_numpy(), dtype=torch.float32)
    if p.numel():
        print(
            f"[prob] score=p_triage min={float(p.min()):.6f} mean={float(p.mean()):.6f} "
            f"max={float(p.max()):.6f} pred_pos_rate={metrics['pred_pos_rate']:.4f}",
            flush=True,
        )
    if abs(metrics["AUC"] - args.expected_auc) > args.warn_tol or abs(metrics["AUPRC"] - args.expected_auprc) > args.warn_tol or abs(metrics["MCC"] - args.expected_mcc) > args.warn_tol:
        print("[WARN] current diagnostic run does not match expected benchmark; check checkpoint/split/data config.", flush=True)
    if args.export_fusion_diagnostics:
        tuna_write_tsv(eval_df, args.out_tsv)
    else:
        rows = []
        for _, row in eval_df.iterrows():
            rows.append({
                "pair_id": row["pair_id"],
                "protein_A": row["protein_A"],
                "protein_B": row["protein_B"],
                "label": int(row["label"]),
                "prob": float(row["p_triage"]),
                "pred": int(row["pred_auto"]),
            })
        write_pair_rows(args.out_tsv, rows)
    if args.out_tsv:
        print(f"[predictions] {args.out_tsv}", flush=True)


def write_pair_rows(path: str, rows: List[Dict]):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["pair_id", "protein_A", "protein_B", "label", "prob", "pred"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser(description="Predict or evaluate PertiNet on residue-level datasets or TUnA pair data.")
    p.add_argument("--dataset", choices=["pp", "gpsite", "dest", "rbp400", "tuna_pair"], default="rbp400")
    p.add_argument("--checkpoint", default="", help="Default: GPSite balanced checkpoint for --dataset gpsite; PP checkpoint for --dataset pp.")
    p.add_argument("--pp-root", default="", help="Override GraphRBF-PP prepared root.")
    p.add_argument("--gpsite-root", default="", help="Override GPSite prepared root.")
    p.add_argument("--dest-root", default="", help="Override Dest prepared root.")
    p.add_argument("--rbp400-root", default="", help="Override RBP400 prepared root.")
    p.add_argument("--calibrate-split", default="val", help="Prepared split name, e.g. train/val/test.")
    p.add_argument("--eval-split", default="test", help="Prepared split name, e.g. test or test_dset186.")
    p.add_argument("--split-mode", choices=["random", "group", "protein_pair", "protein_component", "protein_disjoint"], default="protein_disjoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", default="auto", help="'auto' uses calibrate split; otherwise a numeric threshold.")
    p.add_argument("--threshold-policy", choices=["training", "fbeta"], default="training", help="training matches the balanced-v6 threshold objective; fbeta uses only F-beta.")
    p.add_argument("--threshold-beta", type=float, default=-1.0, help="Beta for auto threshold. <=0 uses l1_threshold_beta saved in the checkpoint/config.")
    p.add_argument("--smooth-window", type=int, default=1, help="Odd moving-average window on residue probabilities. 1 disables smoothing.")
    p.add_argument("--peak-boost", type=float, default=0.0, help="Optional local peak boost on residue probabilities after smoothing; 0 disables.")
    p.add_argument("--no-rbp400-ager", action="store_true", help="Disable RBP400 AGER post-processing during prediction.")
    p.add_argument(
        "--checkpoint-preset",
        choices=[
            "paper", "best", "score", "acc", "precision", "recall", "f1", "auroc", "auprc", "mcc",
            "swa", "ensemble_best_swa", "ensemble_best_mcc", "ensemble_best_precision",
            "topk", "topk_engineering", "topk_score", "topk_target", "topk_recall_l5", "topk_recall_l10",
            "topk_precision_10", "topk_p10_r10", "topk_swa", "topk_ensemble_best_swa",
        ],
        default="paper",
        help="RBP400 checkpoint preset used when --checkpoint is empty.",
    )
    p.add_argument("--ensemble-checkpoints", default="", help="Extra comma/semicolon-separated checkpoints to average with --checkpoint.")
    p.add_argument("--rbp400-sweep", action="store_true", help="Evaluate a small RBP400 preset/post-processing grid and export the selected rank TSV.")
    p.add_argument("--sweep-presets", default="paper,best,score,acc,precision,mcc,f1,auroc,auprc,ensemble_best_swa,ensemble_best_precision", help="Comma-separated RBP400 presets for --rbp400-sweep.")
    p.add_argument("--sweep-peak-boosts", default="0,0.03,0.05", help="Comma-separated peak boost values for --rbp400-sweep.")
    p.add_argument("--sweep-smooth-windows", default="1", help="Comma-separated smoothing windows for --rbp400-sweep.")
    p.add_argument(
        "--sweep-selection",
        choices=["score", "acc", "precision", "recall", "f1", "auroc", "auprc", "mcc"],
        default="score",
        help="Metric used to select the exported RBP400 sweep result.",
    )
    p.add_argument("--use-current-config", action="store_true", help="Use config.py instead of the configuration saved inside the checkpoint.")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--device", default="")
    p.add_argument("--out-tsv", default="", help="Optional per-residue prediction TSV path. Default depends on dataset.")
    p.add_argument("--summary-tsv", default="", help="Optional metrics summary TSV path for L1 prediction/evaluation.")
    p.add_argument("--report-only", action="store_true", help="For --dataset tuna_pair, only report split sizes, label ratios, and leakage checks.")
    p.add_argument("--split-file", default="", help="For --dataset tuna_pair, load a fixed split manifest JSON.")
    p.add_argument("--save-split-file", default="", help="For --dataset tuna_pair, save the generated split manifest JSON.")
    p.add_argument("--export-fusion-diagnostics", action="store_true", help="For --dataset tuna_pair, export branch-level fusion diagnostic columns.")
    p.add_argument("--expected-auc", type=float, default=0.735)
    p.add_argument("--expected-auprc", type=float, default=0.742)
    p.add_argument("--expected-mcc", type=float, default=0.413)
    p.add_argument("--warn-tol", type=float, default=0.03)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint:
        if args.dataset == "rbp400":
            args.checkpoint = rbp400_preset_checkpoints(args.checkpoint_preset)
        else:
            args.checkpoint = {"gpsite": DEFAULT_GPSITE_CKPT, "dest": DEFAULT_DEST_CKPT, "tuna_pair": DEFAULT_TUNA_PAIR_CKPT}.get(args.dataset, DEFAULT_PP_CKPT)
    if args.ensemble_checkpoints:
        args.checkpoint = ",".join(split_checkpoint_list(args.checkpoint) + split_checkpoint_list(args.ensemble_checkpoints))
    checkpoint_paths = split_checkpoint_list(args.checkpoint)
    if not checkpoint_paths:
        raise ValueError("No checkpoint path was provided.")
    if not args.out_tsv:
        args.out_tsv = {"gpsite": DEFAULT_GPSITE_PRED_TSV, "dest": DEFAULT_DEST_PRED_TSV, "rbp400": DEFAULT_RBP400_PRED_TSV, "tuna_pair": DEFAULT_TUNA_PAIR_PRED_TSV}.get(args.dataset, DEFAULT_PP_PRED_TSV)
    if not args.summary_tsv and args.dataset == "dest":
        args.summary_tsv = f"{OUTPUT_DIR}/dset_metrics_summary.tsv"

    cfg, cfg_source = config_from_checkpoint(checkpoint_paths[0], dataset=args.dataset, use_current_config=args.use_current_config)
    if args.dataset == "tuna_pair":
        predict_tuna_pair(args, cfg, cfg_source)
        return
    if args.pp_root:
        cfg.pp_root = args.pp_root
    if args.gpsite_root:
        cfg.gpsite_root = args.gpsite_root
    if args.dest_root:
        cfg.dest_root = args.dest_root
    if args.rbp400_root:
        cfg.rbp400_root = args.rbp400_root
    if args.threshold_beta <= 0:
        args.threshold_beta = float(getattr(cfg, "l1_threshold_beta", 1.0))
    batch_size = args.batch_size if args.batch_size > 0 else int(getattr(cfg, "l1_batch_size", getattr(cfg, "batch_size", 1)))
    num_workers = args.num_workers if args.num_workers >= 0 else int(getattr(cfg, "l1_num_workers", getattr(cfg, "num_workers", 0)))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.dataset == "rbp400":
        print(
            f"[config] source={cfg_source} dataset=rbp400 checkpoint={args.checkpoint} "
            f"preset={args.checkpoint_preset} ensemble_n={len(checkpoint_paths)} "
            f"eval={args.eval_split} batch_size={batch_size} workers={num_workers} "
            f"smooth_window={args.smooth_window} peak_boost={args.peak_boost:.3f} "
            f"AGER={not bool(args.no_rbp400_ager)} "
            f"thr_mode={str(getattr(cfg, 'l1_threshold_mode', 'auto_mcc'))} "
            f"thr_min_recall={float(getattr(cfg, 'l1_threshold_min_recall', 0.0)):.2f} "
            f"output=binary+rank-tsv",
            flush=True,
        )
    else:
        print(
            f"[config] source={cfg_source} dataset={args.dataset} checkpoint={args.checkpoint} "
            f"calibrate={args.calibrate_split} eval={args.eval_split} batch_size={batch_size} workers={num_workers} "
            f"smooth_window={args.smooth_window} threshold_policy={args.threshold_policy} beta={args.threshold_beta:.3f} "
            f"thr_floor=(p:{float(getattr(cfg, 'l1_threshold_min_precision', 0.0)):.2f},r:{float(getattr(cfg, 'l1_threshold_min_recall', 0.0)):.2f}) "
            f"refiner={int(bool(getattr(cfg, 'use_l1_residue_refiner', False)))}:{float(getattr(cfg, 'l1_refiner_alpha', 0.0)):.2f} "
            f"geom={int(bool(getattr(cfg, 'use_l1_geom_adapter', False)))}:{float(getattr(cfg, 'l1_geom_alpha', 0.0)):.2f}",
            flush=True,
        )
    print(
        f"[data-root] pp_root={getattr(cfg, 'pp_root', '')} "
        f"gpsite_root={getattr(cfg, 'gpsite_root', '')} "
        f"dest_root={getattr(cfg, 'dest_root', '')} "
        f"rbp400_root={getattr(cfg, 'rbp400_root', '')}",
        flush=True,
    )

    if args.dataset == "rbp400":
        if args.rbp400_sweep:
            presets = parse_csv_values(args.sweep_presets, str)
            boosts = parse_csv_values(args.sweep_peak_boosts, float)
            windows = parse_csv_values(args.sweep_smooth_windows, int)
            best_item = None
            best_rows: List[Dict] = []
            print(
                f"[sweep] presets={','.join(presets)} boosts={','.join(str(x) for x in boosts)} "
                f"windows={','.join(str(x) for x in windows)} selection={args.sweep_selection}",
                flush=True,
            )
            for preset in presets:
                ckpt_value = rbp400_preset_checkpoints(preset)
                model_group, paths = load_model_group(cfg, ckpt_value, device)
                for smooth in windows:
                    for boost in boosts:
                        eval_loader = make_loader(cfg, args.eval_split, batch_size, num_workers, args.dataset)
                        prob, lab, rows = collect_l1_predictions(
                            model_group,
                            eval_loader,
                            device,
                            smooth,
                            cfg=cfg,
                            use_ager=not bool(args.no_rbp400_ager),
                            peak_boost=boost,
                        )
                        metrics = binary_metrics_best_threshold(
                            prob,
                            lab,
                            beta=args.threshold_beta,
                            min_precision=float(getattr(cfg, "l1_threshold_min_precision", 0.0)),
                            min_recall=float(getattr(cfg, "l1_threshold_min_recall", 0.0)),
                            mode=str(getattr(cfg, "l1_threshold_mode", "auto_mcc")),
                        )
                        topk_metrics = topk_metrics_from_rows(rows)
                        score = rbp400_weighted_score(metrics)
                        select_value = rbp400_sweep_sort_key(metrics, args.sweep_selection)
                        print(
                            f"[sweep] preset={preset} ensemble_n={len(paths)} smooth={smooth} peak={boost:.3f} "
                            f"ACC={metrics['acc']:.4f} Precision={metrics['precision']:.4f} "
                            f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
                            f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
                            f"MCC={metrics['mcc']:.4f} score={score:.4f} select={select_value:.4f} "
                            f"topk(R@L/10={topk_metrics['l1_recall_l10']:.4f},P@10={topk_metrics['l1_precision_10']:.4f})",
                            flush=True,
                        )
                        if best_item is None or select_value > best_item["select"]:
                            best_item = {
                                "preset": preset,
                                "paths": paths,
                                "smooth": smooth,
                                "boost": boost,
                                "metrics": metrics,
                                "topk": topk_metrics,
                                "score": score,
                                "select": select_value,
                            }
                            best_rows = rows
                if isinstance(model_group, list):
                    del model_group
                else:
                    del model_group
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if best_item is None:
                raise RuntimeError("RBP400 sweep produced no candidates.")
            m = best_item["metrics"]
            topk_m = best_item["topk"]
            print(
                f"[sweep-best] preset={best_item['preset']} smooth={best_item['smooth']} peak={best_item['boost']:.3f} "
                f"ACC={m['acc']:.4f} Precision={m['precision']:.4f} Recall={m['recall']:.4f} "
                f"F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
                f"MCC={m['mcc']:.4f} score={best_item['score']:.4f} "
                f"topk(R@L/10={topk_m['l1_recall_l10']:.4f},P@10={topk_m['l1_precision_10']:.4f})",
                flush=True,
            )
            print_rbp400_bucket_metrics(cfg, f"sweep-best/{args.eval_split}", best_rows, float(m["thr"]))
            write_rank_rows(args.out_tsv, best_rows, threshold=float(best_item["metrics"]["thr"]))
            if args.out_tsv:
                print(f"[rank-predictions] {args.out_tsv}", flush=True)
            return

        model_or_models, _ = load_model_group(cfg, args.checkpoint, device)
        eval_loader = make_loader(cfg, args.eval_split, batch_size, num_workers, args.dataset)
        prob, lab, rows = collect_l1_predictions(
            model_or_models,
            eval_loader,
            device,
            args.smooth_window,
            cfg=cfg,
            use_ager=not bool(args.no_rbp400_ager),
            peak_boost=args.peak_boost,
        )
        bm = binary_metrics_best_threshold(
            prob,
            lab,
            beta=args.threshold_beta,
            min_precision=float(getattr(cfg, "l1_threshold_min_precision", 0.0)),
            min_recall=float(getattr(cfg, "l1_threshold_min_recall", 0.0)),
            mode=str(getattr(cfg, "l1_threshold_mode", "auto_mcc")),
        )
        print_metrics(f"eval/{args.eval_split}", bm, float(bm["thr"]))
        print_topk_metrics(f"eval/{args.eval_split}", topk_metrics_from_rows(rows))
        print_rbp400_bucket_metrics(cfg, f"eval/{args.eval_split}", rows, float(bm["thr"]))
        write_rank_rows(args.out_tsv, rows, threshold=float(bm["thr"]))
        if args.out_tsv:
            print(f"[rank-predictions] {args.out_tsv}", flush=True)
        return

    model_or_models, _ = load_model_group(cfg, args.checkpoint, device)
    cal_loader = make_loader(cfg, args.calibrate_split, batch_size, num_workers, args.dataset)
    cal_prob, cal_y, _ = collect_l1_predictions(model_or_models, cal_loader, device, args.smooth_window, peak_boost=args.peak_boost)
    if args.threshold == "auto":
        objective_weights = metric_objective_weights(cfg) if args.threshold_policy == "training" else None
        balance_w = float(getattr(cfg, "l1_threshold_balance_w", 0.0)) if args.threshold_policy == "training" else 0.0
        min_precision = float(getattr(cfg, "l1_threshold_min_precision", 0.0)) if args.threshold_policy == "training" else 0.0
        min_recall = float(getattr(cfg, "l1_threshold_min_recall", 0.0)) if args.threshold_policy == "training" else 0.0
        cal_metrics = binary_metrics_best_threshold(
            cal_prob,
            cal_y,
            beta=args.threshold_beta,
            objective_weights=objective_weights,
            balance_w=balance_w,
            min_precision=min_precision,
            min_recall=min_recall,
            mode=str(getattr(cfg, "l1_threshold_mode", "auto_mcc")),
        )
        threshold = float(cal_metrics["thr"])
        print_metrics(f"calibrate/{args.calibrate_split}", cal_metrics, threshold)
    else:
        threshold = float(args.threshold)
        cal_metrics = binary_metrics(cal_prob, cal_y, thr=threshold)
        print_metrics(f"calibrate/{args.calibrate_split}", cal_metrics, threshold)

    eval_loader = make_loader(cfg, args.eval_split, batch_size, num_workers, args.dataset)
    eval_prob, eval_y, rows = collect_l1_predictions(model_or_models, eval_loader, device, args.smooth_window, peak_boost=args.peak_boost)
    eval_metrics = binary_metrics(eval_prob, eval_y, thr=threshold)
    topk_metrics = topk_metrics_from_rows(rows)
    print_metrics(f"eval/{args.eval_split}", eval_metrics, threshold)
    print_topk_metrics(f"eval/{args.eval_split}", topk_metrics)
    write_rows(args.out_tsv, rows, threshold)
    if args.summary_tsv:
        summary_rows = []
        for tag, metrics in ((f"calibrate/{args.calibrate_split}", cal_metrics), (f"eval/{args.eval_split}", eval_metrics)):
            row = {"tag": tag, "threshold": threshold}
            row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            if tag.startswith("eval/"):
                row.update({k: float(v) for k, v in topk_metrics.items()})
            summary_rows.append(row)
        write_l1_summary(args.summary_tsv, summary_rows)
        print(f"[summary] {args.summary_tsv}", flush=True)
    if args.out_tsv:
        print(f"[predictions] {args.out_tsv}", flush=True)


if __name__ == "__main__":
    main()
