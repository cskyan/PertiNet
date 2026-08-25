#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export pair-level PertiNet evidence for the RBP400 case study.

This script is the internal standard-table exporter. It reads canonical RBP400
candidate pairs and writes `triage_pair_outputs.tsv` with p_TRIAGE, L1/L2/L3
fusion weights, Top-L/10 residue hotspot summaries, dynamic Top-M interface
support, and placeholders for later occlusion/ESI merging.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from predict import rbp400_preset_checkpoints
from train import (
    batch_to_device,
    collate_pad,
    find_l1_feature_file,
    forward_batch,
    load_esm_map,
    load_l1_array,
    load_l1_coords,
    load_resume,
    pad_residue_arrays,
    truncate_feature,
)
from config import rbp400_triage_data_config
from model import TRIAGEPPIModel


OUT_FIELDS = [
    "pair_id",
    "protein_A",
    "protein_B",
    "gene_A",
    "gene_B",
    "triage_score",
    "logit_TRIAGE",
    "p_TRIAGE",
    "g_L1",
    "g_L2",
    "g_L3",
    "p_pair_raw",
    "L1_hotspot_score_A",
    "L1_hotspot_score_B",
    "L1_hotspot_score_pair",
    "L2_interface_score",
    "L2_interface_max",
    "L2_interface_density_topM",
    "topK_residues_A",
    "topK_residues_B",
    "topM_pairs",
    "valid_pair_count",
    "disease_pair_category",
    "occlusion_drop_L1",
    "occlusion_drop_L2",
    "occlusion_drop_random",
    "occlusion_drop_domain_random",
    "occlusion_drop_mean_L1L2",
    "ESI",
]


DEFAULT_RUN_DIR = Path("checkpoints")


def count_physical_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return sum(1 for _ in handle)


def backup_existing_output(path: Path) -> None:
    line_count = count_physical_lines(path)
    if line_count <= 1:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(str(path), str(backup))
    print(f"[case-study-export] backup_existing={backup} lines={line_count}", flush=True)


def available_checkpoints(run_dir: Path = DEFAULT_RUN_DIR) -> List[str]:
    if not run_dir.exists():
        return []
    names = []
    for path in sorted(run_dir.glob("*.pt")):
        name = path.name
        if "rbp400" in name.lower() or "topk" in name.lower():
            names.append(str(path))
    return names


def resolve_checkpoint(args) -> str:
    checkpoint = args.checkpoint.strip()
    if checkpoint:
        if "your" in checkpoint.lower():
            choices = available_checkpoints()
            detail = "\n".join(f"  {p}" for p in choices[:20]) if choices else f"  no .pt files found under {DEFAULT_RUN_DIR}"
            raise SystemExit(
                "Do not pass the placeholder checkpoint path. Either omit --checkpoint to use --preset triage, "
                "or pass a real .pt file.\nAvailable checkpoint candidates:\n" + detail
            )
        if not Path(checkpoint).exists():
            choices = available_checkpoints()
            detail = "\n".join(f"  {p}" for p in choices[:20]) if choices else f"  no .pt files found under {DEFAULT_RUN_DIR}"
            raise SystemExit(f"Checkpoint not found: {checkpoint}\nAvailable checkpoint candidates:\n{detail}")
        return checkpoint
    preset_checkpoint = rbp400_preset_checkpoints(args.preset)
    first = preset_checkpoint.split(",")[0]
    if not Path(first).exists():
        choices = available_checkpoints()
        detail = "\n".join(f"  {p}" for p in choices[:20]) if choices else f"  no .pt files found under {DEFAULT_RUN_DIR}"
        raise SystemExit(
            f"Preset --preset {args.preset} resolved to a missing checkpoint: {first}\n"
            f"Pass --checkpoint with one of these real files:\n{detail}"
        )
    return preset_checkpoint


def canonical_pair(a: str, b: str) -> Tuple[str, str, str]:
    aa = str(a).strip()
    bb = str(b).strip()
    if not aa or not bb:
        raise ValueError(f"empty pair endpoint: {a!r}, {b!r}")
    left, right = sorted([aa, bb])
    return left, right, f"{left}__{right}"


def read_candidate_pairs(path: Path, limit: int = 0) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_A", "protein_B"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows: List[Dict[str, str]] = []
        seen = set()
        for row in reader:
            a, b, pair_id = canonical_pair(row.get("protein_A", ""), row.get("protein_B", ""))
            if pair_id in seen:
                continue
            seen.add(pair_id)
            out = dict(row)
            swap = a != str(row.get("protein_A", "")).strip()
            out["protein_A"] = a
            out["protein_B"] = b
            out["pair_id"] = pair_id
            if swap:
                out["gene_A"], out["gene_B"] = row.get("gene_B", ""), row.get("gene_A", "")
            else:
                out["gene_A"], out["gene_B"] = row.get("gene_A", ""), row.get("gene_B", "")
            rows.append(out)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


class RBP400FeatureStore:
    def __init__(self, cfg, root: Path, esm_dir: Path):
        self.cfg = cfg
        self.root = root
        self.esm = load_esm_map(esm_dir, max_len=int(getattr(cfg, "max_site_len", 0))) if bool(getattr(cfg, "use_esm", True)) else {}
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _npy(self, subdir: str, pid: str) -> Optional[np.ndarray]:
        aliases = {
            "labels": ("labels", "label", "annotations", "annotations_case_realdata"),
            "pssm": ("pssm", "PSSM"),
            "dssp": ("dssp", "DSSP"),
            "coords": ("coords", "structures", "structure", "coordinates"),
        }
        path = find_l1_feature_file(self.root, aliases.get(subdir, (subdir,)), pid)
        if path is None:
            return None
        return load_l1_array(path, subdir).astype(np.float32)

    def get(self, pid: str) -> Dict[str, torch.Tensor]:
        if pid in self.cache:
            return self.cache[pid]
        y = self._npy("labels", pid)
        if y is None:
            esm_arr = self.esm.get(pid)
            if esm_arr is None:
                raise FileNotFoundError(f"Cannot infer length: missing labels and ESM for {pid} under {self.root}")
            L = int(esm_arr.shape[0])
        else:
            y = np.asarray(y, dtype=np.float32).reshape(-1)
            L = int(y.shape[0])
        pssm = self._npy("pssm", pid)
        dssp = self._npy("dssp", pid)
        esm_arr = self.esm.get(pid)
        esm_placeholder = np.zeros((L, 1280), dtype=np.float32) if bool(getattr(self.cfg, "use_esm", True)) and esm_arr is None else None
        res = truncate_feature(
            pad_residue_arrays([esm_arr, esm_placeholder, pssm, dssp], int(self.cfg.d_res_in), length=L),
            int(getattr(self.cfg, "max_site_len", 1024)),
        )
        coords = load_l1_coords(self.root, pid)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            coord_arr = np.full((res.size(0), 3), np.nan, dtype=np.float32)
            if coords.ndim == 2 and coords.shape[1] >= 3:
                n = min(int(coords.shape[0]), int(res.size(0)))
                if n > 0:
                    coord_arr[:n, :3] = coords[:n, :3].astype(np.float32)
        else:
            coord_arr = np.full((res.size(0), 3), np.nan, dtype=np.float32)
        item = {
            "res": res,
            "mask": torch.ones(res.size(0), dtype=torch.float32),
            "coords": torch.from_numpy(coord_arr),
        }
        self.cache[pid] = item
        return item


class RBP400CasePairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], store: RBP400FeatureStore):
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        a = row["protein_A"]
        b = row["protein_B"]
        fa = self.store.get(a)
        fb = self.store.get(b)
        return {
            "pair_id": row["pair_id"],
            "protein_A": a,
            "protein_B": b,
            "gene_A": row.get("gene_A", ""),
            "gene_B": row.get("gene_B", ""),
            "disease_pair_category": row.get("disease_pair_category", ""),
            "resA": fa["res"],
            "resB": fb["res"],
            "coordsA": fa["coords"],
            "coordsB": fb["coords"],
            "maskA": fa["mask"],
            "maskB": fb["mask"],
            "y_pair": torch.tensor(0.0, dtype=torch.float32),
        }


def case_collate(batch: List[Dict]) -> Dict:
    meta_keys = ("gene_A", "gene_B", "disease_pair_category")
    tensor_batch = [{k: v for k, v in row.items() if k not in meta_keys} for row in batch]
    out = collate_pad(tensor_batch)
    for key in meta_keys:
        out[key] = [row.get(key, "") for row in batch]
    return out


def top_fraction_indices(prob: torch.Tensor, mask: torch.Tensor, frac: float) -> torch.Tensor:
    valid = torch.nonzero(mask > 0.5, as_tuple=False).flatten()
    if valid.numel() == 0:
        return valid
    k = max(1, int(math.ceil(float(valid.numel()) * float(frac))))
    vals = prob[valid]
    order = torch.argsort(vals, descending=True)[:k]
    return valid[order]


def format_residue_list(prob: torch.Tensor, idx: torch.Tensor) -> str:
    parts = []
    for i in idx.detach().cpu().tolist():
        parts.append(f"{int(i) + 1}:{float(prob[int(i)].detach().cpu()):.6g}")
    return ";".join(parts)


def dynamic_topm_count(valid_pair_count: int) -> int:
    return min(128, max(20, int(round(0.02 * float(valid_pair_count)))))


def summarize_topm(p_interface: torch.Tensor, maskA: torch.Tensor, maskB: torch.Tensor) -> Tuple[float, float, float, str, int]:
    valid2d = (maskA > 0.5).unsqueeze(1) & (maskB > 0.5).unsqueeze(0)
    valid_pair_count = int(valid2d.sum().item())
    if valid_pair_count <= 0:
        return 0.0, 0.0, 0.0, "", 0
    vals = p_interface[valid2d]
    m = min(dynamic_topm_count(valid_pair_count), int(vals.numel()))
    top_vals, flat_order = torch.topk(vals, m, largest=True, sorted=True)
    coords = torch.nonzero(valid2d, as_tuple=False)[flat_order]
    mean_topm = float(top_vals.mean().detach().cpu())
    max_topm = float(top_vals.max().detach().cpu())
    density = float((top_vals >= 0.5).float().mean().detach().cpu())
    parts = []
    for coord, val in zip(coords.detach().cpu().tolist(), top_vals.detach().cpu().tolist()):
        parts.append(f"{int(coord[0]) + 1}-{int(coord[1]) + 1}:{float(val):.6g}")
    return mean_topm, max_topm, density, ";".join(parts), valid_pair_count


def fmt(value: float) -> str:
    return f"{float(value):.8g}"


@torch.no_grad()
def export_rows(
    model,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    l1_top_frac: float,
    progress_every: int,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=OUT_FIELDS)
        writer.writeheader()
        for batch in loader:
            pair_ids = list(batch["pair_id"])
            protein_a = list(batch["protein_A"])
            protein_b = list(batch["protein_B"])
            gene_a = list(batch.get("gene_A", [""] * len(pair_ids)))
            gene_b = list(batch.get("gene_B", [""] * len(pair_ids)))
            disease_cat = list(batch.get("disease_pair_category", [""] * len(pair_ids)))
            batch_dev = batch_to_device({k: v for k, v in batch.items() if k not in {"gene_A", "gene_B", "disease_pair_category"}}, device)
            out = forward_batch(model, batch_dev)
            gates = out["fusion_weights"]
            for i, pair_id in enumerate(pair_ids):
                p_res_a = out["p_res_A"][i]
                p_res_b = out["p_res_B"][i]
                mask_a = batch_dev["maskA"][i]
                mask_b = batch_dev["maskB"][i]
                idx_a = top_fraction_indices(p_res_a, mask_a, l1_top_frac)
                idx_b = top_fraction_indices(p_res_b, mask_b, l1_top_frac)
                score_a = float(p_res_a[idx_a].mean().detach().cpu()) if idx_a.numel() else 0.0
                score_b = float(p_res_b[idx_b].mean().detach().cpu()) if idx_b.numel() else 0.0
                l2_mean, l2_max, l2_density, topm_pairs, valid_pair_count = summarize_topm(
                    out["p_interface"][i],
                    mask_a,
                    mask_b,
                )
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "protein_A": protein_a[i],
                        "protein_B": protein_b[i],
                        "gene_A": gene_a[i],
                        "gene_B": gene_b[i],
                        "triage_score": fmt(float(out["logit_triage"][i].detach().cpu())),
                        "logit_TRIAGE": fmt(float(out["logit_triage"][i].detach().cpu())),
                        "p_TRIAGE": fmt(float(out["p_triage"][i].detach().cpu())),
                        "g_L1": fmt(float(gates["g_res"][i].detach().cpu())),
                        "g_L2": fmt(float(gates["g_interface"][i].detach().cpu())),
                        "g_L3": fmt(float(gates["g_pair"][i].detach().cpu())),
                        "p_pair_raw": fmt(float(out["p_pair_raw"][i].detach().cpu())),
                        "L1_hotspot_score_A": fmt(score_a),
                        "L1_hotspot_score_B": fmt(score_b),
                        "L1_hotspot_score_pair": fmt((score_a + score_b) / 2.0),
                        "L2_interface_score": fmt(l2_mean),
                        "L2_interface_max": fmt(l2_max),
                        "L2_interface_density_topM": fmt(l2_density),
                        "topK_residues_A": format_residue_list(p_res_a, idx_a),
                        "topK_residues_B": format_residue_list(p_res_b, idx_b),
                        "topM_pairs": topm_pairs,
                        "valid_pair_count": str(valid_pair_count),
                        "disease_pair_category": disease_cat[i],
                        "occlusion_drop_L1": "",
                        "occlusion_drop_L2": "",
                        "occlusion_drop_random": "",
                        "occlusion_drop_domain_random": "",
                        "occlusion_drop_mean_L1L2": "",
                        "ESI": "",
                    }
                )
                written += 1
                if progress_every > 0 and written % progress_every == 0:
                    print(f"[case-study-export] written={written}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RBP400 case-study pair evidence table.")
    parser.add_argument("--candidate-pairs", default="case_study_standard/tables/rbp400_candidate_pairs.tsv")
    parser.add_argument("--out-tsv", default="case_study_standard/tables/triage_pair_outputs.tsv")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. If empty, use --preset.")
    parser.add_argument(
        "--preset",
        default="triage",
        choices=[
            "triage",
            "triage_score",
            "triage_target",
            "triage_recall_l5",
            "triage_recall_l10",
            "triage_precision_10",
            "triage_p10_r10",
            "triage_swa",
            "triage_ensemble_best_swa",
            "topk",
            "topk_engineering",
            "topk_score",
            "topk_target",
            "topk_recall_l5",
            "topk_recall_l10",
            "topk_precision_10",
            "topk_p10_r10",
            "topk_swa",
            "topk_ensemble_best_swa",
        ],
    )
    parser.add_argument("--rbp400-root", default="")
    parser.add_argument("--rbp400-esm-dir", default="")
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses rbp400_triage_data_config().batch_size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--l1-top-frac", type=float, default=0.10, help="Top-L fraction for L1 hotspot score; default is Top-L/10.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args)
    cfg = rbp400_triage_data_config()
    cfg_source = "rbp400_triage_data_config(current; checkpoint config intentionally ignored)"
    if args.rbp400_root:
        cfg.rbp400_root = args.rbp400_root
    if args.rbp400_esm_dir:
        cfg.rbp400_esm_dir = args.rbp400_esm_dir
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg.batch_size)
    cfg.l1_exclude_zero_label_proteins = False

    rows = read_candidate_pairs(Path(args.candidate_pairs), limit=args.limit)
    if not rows:
        raise SystemExit(f"No candidate pairs found in {args.candidate_pairs}. Fill rbp400_master_annotation.tsv and run build_candidate_pairs.py first.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[case-study-export] pairs={len(rows)} batch_size={batch_size} device={device} "
        f"checkpoint={checkpoint.split(',')[0]}",
        flush=True,
    )
    store = RBP400FeatureStore(cfg, Path(cfg.rbp400_root), Path(cfg.rbp400_esm_dir))
    ds = RBP400CasePairDataset(rows, store)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=case_collate, num_workers=args.num_workers)
    model = TRIAGEPPIModel(cfg).to(device)
    load_resume(model, checkpoint.split(",")[0], device)
    model.eval()
    out_path = Path(args.out_tsv)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    backup_existing_output(out_path)
    written = export_rows(model, loader, device, tmp_path, float(args.l1_top_frac), int(args.progress_every))
    if written != len(rows):
        raise RuntimeError(f"incomplete export: written={written} expected={len(rows)} tmp={tmp_path}")
    tmp_path.replace(out_path)
    print(
        f"[case-study-export] rows={written} out={args.out_tsv} checkpoint={checkpoint.split(',')[0]} "
        f"config={cfg_source} pair_id=minUniProt__maxUniProt triage_score=logit_TRIAGE L1=meanTopL10 L2=meanDynamicTopM",
        flush=True,
    )


if __name__ == "__main__":
    main()
