# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import itertools
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import current_data_config, dest_data_config, dest_triage_config, feature_spec, rbp400_data_config, rbp400_topk_data_config, rbp400_triage_data_config, topk_engineering_data_config, tuna_pair_finetune_config
from losses import compute_joint_loss
from model import TRIAGEConfig, TRIAGEPPIModel


try:
    torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


STAGE_CKPT = {
    "debug": "triage_debug.pt",
    "struct_pretrain": "triage_stage1_struct.pt",
    "pair_fusion": "triage_stage2_pair.pt",
    "joint_finetune": "triage_final.pt",
    "l1_dest": "triage_l1_dest.pt",
    "dest_triage": "triage_dest_triage.pt",
    "l1_graphrbf": "triage_l1_dest.pt",
    "l1_rbp400": "triage_l1_rbp400_binary.pt",
    "l1_rbp400_topk": "triage_l1_rbp400_topk.pt",
    "rbp400_triage": "triage_rbp400_triage.pt",
    "tuna_pair_finetune": "triage_tuna_pair_finetuned.pt",
}
TOPK_ENGINEERING_CKPT = {
    "struct_pretrain": "triage_topk_stage1_struct.pt",
    "pair_fusion": "triage_topk_stage2_pair.pt",
    "joint_finetune": "triage_topk_final.pt",
}
ENGINEERING_STAGES = ("struct_pretrain", "pair_fusion", "joint_finetune")
RBP400_L1_STAGES = ("l1_rbp400", "l1_rbp400_topk")
RBP400_TRIAGE_STAGES = ("rbp400_triage",)
TUNA_PAIR_STAGES = ("tuna_pair_finetune",)
DEST_TRIAGE_STAGES = ("dest_triage",)
RBP400_STAGES = RBP400_L1_STAGES + RBP400_TRIAGE_STAGES
TOPK_STAGES = ("l1_rbp400_topk",)
L1_STAGES = ("l1_dest", "l1_graphrbf") + RBP400_L1_STAGES

DEFAULT_ROOT = "data"
DEFAULT_OUT_DIR = "outputs"
DEFAULT_PP_ROOT = f"{DEFAULT_ROOT}/Dset_prepared"
DEFAULT_RBP400_ROOT = f"{DEFAULT_ROOT}/RBP400"
DEFAULT_RBP400_ID_LIST = f"{DEFAULT_ROOT}/RBP400/accessions.txt"
DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D.pt"
AA20 = list("ARNDCQEGHILKMFPSTWYV")
SS8 = ["H", "G", "I", "E", "B", "T", "S", "C"]
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "C",
    "PYL": "K",
    "MSE": "M",
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticTRIAGEDataset(Dataset):
    def __init__(self, n: int, d_res: int, la: int = 48, lb: int = 52, task: str = "struct"):
        self.n = int(n)
        self.d_res = int(d_res)
        self.la = int(la)
        self.lb = int(lb)
        self.task = task

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> Dict:
        gen = torch.Generator().manual_seed(1000 + idx + (0 if self.task == "struct" else 100000))
        resA = torch.randn(self.la, self.d_res, generator=gen)
        resB = torch.randn(self.lb, self.d_res, generator=gen)
        coordsA = torch.cumsum(torch.randn(self.la, 3, generator=gen), dim=0)
        coordsB = torch.cumsum(torch.randn(self.lb, 3, generator=gen), dim=0) + 4.0
        maskA = torch.ones(self.la)
        maskB = torch.ones(self.lb)
        signal = (resA[:, :8].mean(-1).unsqueeze(1) + resB[:, :8].mean(-1).unsqueeze(0))
        geom = 8.0 - torch.cdist(coordsA, coordsB)
        y2d = ((signal + 0.2 * geom) > 1.2).float()
        if y2d.sum() == 0 and idx % 2 == 0:
            y2d[idx % self.la, (idx * 7) % self.lb] = 1.0
        y_res_A = y2d.max(dim=1).values
        y_res_B = y2d.max(dim=0).values
        y_pair = (y2d.sum() > 0).float()
        if self.task == "pair":
            y_pair = ((resA[:, :4].mean() + resB[:, :4].mean()) > 0.0).float()
        return {
            "protein_A": f"A{idx}",
            "protein_B": f"B{idx}",
            "resA": resA,
            "resB": resB,
            "maskA": maskA,
            "maskB": maskB,
            "coordsA": coordsA,
            "coordsB": coordsB,
            "y2d": y2d,
            "y_res_A": y_res_A,
            "y_res_B": y_res_B,
            "y_pair": y_pair,
        }


def read_table(path: Path) -> pd.DataFrame:
    first = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                first = line.strip()
                break
    if first.startswith("#") and "\t" not in first:
        header = first.lstrip("#").strip().split()
        return pd.read_csv(path, sep=r"\s+", comment="#", names=header, engine="python")
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1 and " " in str(df.columns[0]):
        return pd.read_csv(path, sep=r"\s+", engine="python")
    return df


def parse_fasta_lengths(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    cur = None
    seq = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = len("".join(seq))
                cur = line[1:].split()[0]
                seq = []
            else:
                seq.append(re.sub("[^A-Za-z]", "", line))
        if cur is not None:
            out[cur] = len("".join(seq))
    return out


def count_npy_files(path: Path) -> int:
    return len(list(path.glob("*.npy"))) if path.exists() else 0


def find_site_fasta(cfg: TRIAGEConfig) -> Optional[Path]:
    candidates = [
        Path(cfg.site_global_dir) / "seq.all.fasta",
        Path(cfg.site_global_dir) / "seq.fasta",
        Path(cfg.site_global_dir) / "all.fasta",
        Path(cfg.site_global_dir) / "proteins.fasta",
        Path(cfg.site_global_dir) / "site.seq.fasta",
        Path(cfg.site_global_dir) / "site.fasta",
        Path(cfg.site_global_dir) / "sequences.fasta",
        Path(cfg.site_global_dir) / "proteins.fa",
        Path(cfg.project_root) / "site_data" / "proteins.fasta",
    ]
    for path in candidates:
        if path.exists() and len(parse_fasta_lengths(path)) > 0:
            return path
    return None


def residue_to_aa1(value) -> str:
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE", "NA", "-"}:
        return "X"
    s = re.sub("[^A-Z]", "", s)
    if len(s) == 1:
        return s if s in AA1 else "X"
    if len(s) == 3:
        return AA3_TO_1.get(s, "X")
    return "X"


def write_wrapped_fasta_line(f, seq: str, width: int = 80):
    for i in range(0, len(seq), width):
        f.write(seq[i : i + width] + "\n")


def read_fasta_sequence(path: Path) -> str:
    seq = []
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(">"):
                continue
            seq.append(re.sub("[^A-Za-z]", "", s).upper())
    return "".join(seq)


def build_pp_fasta(pp_root: str, out_fasta: Path) -> Optional[Path]:
    root = Path(pp_root)
    seq_dir = root / "seq"
    if not seq_dir.exists():
        print(f"[pp-fasta][warn] seq directory missing: {seq_dir}", flush=True)
        return None
    pids = []
    for name in ("all_ids.txt", "train.txt", "val.txt", "test.txt"):
        path = root / name
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            pids.extend([line.strip() for line in f if line.strip()])
    if not pids:
        pids = [p.stem for p in sorted(seq_dir.glob("*.fasta"))]
    seen = set()
    ordered = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    residues = 0
    with open(out_fasta, "w", encoding="utf-8") as out:
        for pid in ordered:
            path = seq_dir / f"{pid}.fasta"
            if not path.exists():
                alt = seq_dir / f"{pid.lower()}.fasta"
                path = alt if alt.exists() else path
            seq = read_fasta_sequence(path)
            if not seq:
                continue
            out.write(f">{pid}\n")
            write_wrapped_fasta_line(out, seq)
            n += 1
            residues += len(seq)
    print(f"[pp-fasta] wrote={out_fasta} proteins={n} residues={residues}", flush=True)
    return out_fasta if n > 0 else None


def infer_pdb_chain(uid: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [p for p in re.split(r"[_:\-\s]+", str(uid)) if p]
    pdb_id = None
    chain_id = None
    for p in parts:
        if len(p) == 4 and re.match(r"^[A-Za-z0-9]{4}$", p):
            pdb_id = p.lower()
            break
    if pdb_id is not None:
        for p in parts:
            if p.lower() != pdb_id and len(p) <= 3:
                chain_id = p
                break
    if pdb_id is None and len(str(uid)) >= 5 and re.match(r"^[A-Za-z0-9]{4}", str(uid)):
        pdb_id = str(uid)[:4].lower()
        rest = re.sub(r"^[A-Za-z0-9]{4}[_:\-]?", "", str(uid))
        if rest:
            chain_id = rest[:1]
    return pdb_id, chain_id


def open_text_maybe_gzip(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def is_structure_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz", ".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def structure_stem(path: Path) -> str:
    name = path.name
    for suffix in (".pdb.gz", ".ent.gz", ".cif.gz", ".mmcif.gz", ".pdb", ".ent", ".cif", ".mmcif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_pdb_sequences(path: Path) -> Dict[str, str]:
    chains: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    seen = set()
    with open_text_maybe_gzip(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resn = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resid = line[22:27].strip() + line[27:28].strip()
            key = (chain, resid)
            if key in seen:
                continue
            seen.add(key)
            chains[chain].append((resid, AA3_TO_1.get(resn, "X")))
    return {chain: "".join(aa for _, aa in vals) for chain, vals in chains.items() if vals}


def parse_mmcif_sequences(path: Path) -> Dict[str, str]:
    chains: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    seen = set()
    with open_text_maybe_gzip(path) as f:
        lines = iter(f)
        for line in lines:
            if line.strip() != "loop_":
                continue
            cols = []
            data = []
            for raw in lines:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("_"):
                    cols.append(s.split()[0])
                    continue
                if not cols:
                    break
                if not any(c.startswith("_atom_site.") for c in cols):
                    break
                if s.startswith("#"):
                    break
                data.append(raw)
                break
            if not cols or not any(c.startswith("_atom_site.") for c in cols):
                continue
            for raw in itertools.chain(data, lines):
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#") or s == "loop_" or s.startswith("_"):
                    break
                try:
                    parts = shlex.split(s)
                except ValueError:
                    parts = s.split()
                if len(parts) < len(cols):
                    continue
                row = {c: parts[i] for i, c in enumerate(cols)}
                group = row.get("_atom_site.group_PDB", "")
                atom = row.get("_atom_site.auth_atom_id") or row.get("_atom_site.label_atom_id") or ""
                if group != "ATOM" or atom.strip().upper() != "CA":
                    continue
                resn = row.get("_atom_site.auth_comp_id") or row.get("_atom_site.label_comp_id") or ""
                chain = row.get("_atom_site.auth_asym_id") or row.get("_atom_site.label_asym_id") or "_"
                resid = row.get("_atom_site.auth_seq_id") or row.get("_atom_site.label_seq_id") or ""
                ins = row.get("_atom_site.pdbx_PDB_ins_code", "")
                if ins in {".", "?"}:
                    ins = ""
                key = (chain, resid + ins)
                if key in seen:
                    continue
                seen.add(key)
                chains[chain].append((resid + ins, AA3_TO_1.get(resn.upper(), "X")))
    return {chain: "".join(aa for _, aa in vals) for chain, vals in chains.items() if vals}


def parse_structure_sequences(path: Path) -> Dict[str, str]:
    name = path.name.lower()
    if name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return parse_mmcif_sequences(path)
    return parse_pdb_sequences(path)


def build_structure_sequence_index(cfg: TRIAGEConfig) -> Dict[str, str]:
    roots = [
        Path(cfg.site_global_dir),
        Path(cfg.site_homo_dir),
        Path(cfg.site_hetero_dir),
    ]
    seqs: Dict[str, str] = {}
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not is_structure_file(path):
                continue
            scanned += 1
            try:
                chain_seqs = parse_structure_sequences(path)
            except Exception:
                continue
            stem = structure_stem(path)
            pdb_id, _ = infer_pdb_chain(stem)
            base_ids = [stem]
            if pdb_id:
                base_ids.append(pdb_id)
                base_ids.append(pdb_id.upper())
            for chain, seq in chain_seqs.items():
                if not seq:
                    continue
                for base in base_ids:
                    seqs.setdefault(f"{base}_{chain}", seq)
                    seqs.setdefault(f"{base}:{chain}", seq)
                    seqs.setdefault(f"{base}-{chain}", seq)
                if len(chain_seqs) == 1:
                    seqs.setdefault(stem, seq)
    if scanned == 0:
        print(
            f"[site-fasta][warn] scanned SITE structure roots but found no supported structure files: "
            f"{', '.join(str(r) for r in roots)}",
            flush=True,
        )
    else:
        print(f"[site-fasta] scanned supported structure files={scanned} indexed_keys={len(seqs)}", flush=True)
    return seqs


def build_site_fasta_from_structures(cfg: TRIAGEConfig, wanted_uids: List[str], out_fasta: Path) -> Optional[Path]:
    seq_index = build_structure_sequence_index(cfg)
    if not seq_index:
        print("[site-fasta][warn] no supported structure sequences found for SITE fasta fallback.", flush=True)
        return None
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.project_root) / "triage_site_fasta" / "site_fasta_report.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    matched = 0
    residues = 0
    with open(out_fasta, "w", encoding="utf-8") as fasta_f, open(report_path, "w", encoding="utf-8", newline="") as report_f:
        writer = csv.DictWriter(report_f, fieldnames=["uid", "length", "valid_aa", "x_count", "x_rate"], delimiter="\t")
        writer.writeheader()
        for uid in wanted_uids:
            candidates = [uid]
            pdb_id, chain_id = infer_pdb_chain(uid)
            if pdb_id and chain_id:
                candidates.extend([f"{pdb_id}_{chain_id}", f"{pdb_id}:{chain_id}", f"{pdb_id}-{chain_id}", f"{pdb_id.upper()}_{chain_id}"])
            seq = next((seq_index[c] for c in candidates if c in seq_index), "")
            if not seq:
                continue
            valid = sum(1 for aa in seq if aa in AA1)
            if valid == 0:
                continue
            xs = seq.count("X")
            fasta_f.write(f">{uid}\n")
            write_wrapped_fasta_line(fasta_f, seq)
            writer.writerow({"uid": uid, "length": len(seq), "valid_aa": valid, "x_count": xs, "x_rate": xs / max(len(seq), 1)})
            matched += 1
            residues += len(seq)
    print(
        f"[site-fasta] structure fallback matched={matched}/{len(wanted_uids)} wrote={out_fasta} residues={residues}",
        flush=True,
    )
    print(f"[site-fasta] report={report_path}", flush=True)
    return out_fasta if matched > 0 else None


def build_site_fasta_if_possible(cfg: TRIAGEConfig, args) -> Optional[Path]:
    if not getattr(args, "auto_site_fasta", True):
        return None
    out_fasta = Path(cfg.site_global_dir) / "proteins.fasta"
    dssp = Path(cfg.site_global_dir) / "dssp.all.tsv"
    if not dssp.exists():
        print(f"[site-fasta][warn] cannot auto-build SITE fasta; missing DSSP table: {dssp}", flush=True)
        return None
    try:
        df = read_table(dssp)
        uid_col = first_col(df, ["chain_uid", "uid", "protein", "protein_id", "id"])
        aa_col = first_col(df, ["aa", "residue", "res", "resname", "res_name", "amino_acid"])
        idx_col = first_col(df, ["idx", "pos", "res_idx", "residue_index", "resid", "res_id"])
        if uid_col is None or aa_col is None:
            print(
                f"[site-fasta][warn] cannot detect uid/residue columns in {dssp}; "
                f"columns={list(df.columns)[:20]}",
                flush=True,
            )
            if uid_col is None:
                return None
            wanted = sorted(df[uid_col].astype(str).dropna().unique().tolist())
            return build_site_fasta_from_structures(cfg, wanted, out_fasta)
        keep_cols = [uid_col, aa_col] + ([idx_col] if idx_col is not None else [])
        rows = df[keep_cols].copy()
        rows[uid_col] = rows[uid_col].astype(str)
        rows[aa_col] = rows[aa_col].map(residue_to_aa1)
        if idx_col is not None:
            rows["_order"] = pd.to_numeric(rows[idx_col], errors="coerce")
            rows = rows.sort_values([uid_col, "_order"], kind="mergesort")

        out_fasta.parent.mkdir(parents=True, exist_ok=True)
        report_path = Path(cfg.project_root) / "triage_site_fasta" / "site_fasta_report.tsv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        proteins = 0
        residues = 0
        x_count = 0
        with open(out_fasta, "w", encoding="utf-8") as fasta_f, open(report_path, "w", encoding="utf-8", newline="") as report_f:
            writer = csv.DictWriter(report_f, fieldnames=["uid", "length", "valid_aa", "x_count", "x_rate"], delimiter="\t")
            writer.writeheader()
            for uid, group in rows.groupby(uid_col, sort=False):
                seq = "".join(group[aa_col].tolist())
                valid = sum(1 for aa in seq if aa in AA1)
                if valid == 0:
                    continue
                xs = seq.count("X")
                fasta_f.write(f">{uid}\n")
                write_wrapped_fasta_line(fasta_f, seq)
                writer.writerow(
                    {
                        "uid": uid,
                        "length": len(seq),
                        "valid_aa": valid,
                        "x_count": xs,
                        "x_rate": xs / max(len(seq), 1),
                    }
                )
                proteins += 1
                residues += len(seq)
                x_count += xs
        if proteins <= 0:
            print("[site-fasta][warn] DSSP table did not yield usable SITE sequences.", flush=True)
            return None
        print(
            f"[site-fasta] wrote={out_fasta} proteins={proteins} residues={residues} "
            f"x_rate={x_count / max(residues, 1):.4f}",
            flush=True,
        )
        print(f"[site-fasta] report={report_path}", flush=True)
    except Exception as exc:
        print(f"[site-fasta][warn] build failed: {exc}; SITE will use PSSM/DSSP only.", flush=True)
        return None
    return out_fasta if out_fasta.exists() else None


def ensure_esm_cache_for_fasta(
    fasta: Path,
    out_dir: Path,
    local_model: str,
    batch: int,
    max_len: int,
    label: str,
    required_fraction: float = 0.98,
):
    lengths = parse_fasta_lengths(fasta)
    total = len(lengths)
    done = count_npy_files(out_dir)
    if total <= 0:
        print(f"[esm-cache][warn] {label}: no sequences in {fasta}", flush=True)
        return
    need = int(math.ceil(total * float(required_fraction)))
    if done >= need:
        print(f"[esm-cache] {label}: ready {done}/{total} dir={out_dir}", flush=True)
        return
    script = Path(__file__).resolve().parents[1] / "scripts" / "compute_esm.py"
    cmd = [
        sys.executable,
        str(script),
        "--fasta",
        str(fasta),
        "--out",
        str(out_dir),
        "--batch",
        str(int(batch)),
        "--max-len",
        str(int(max_len)),
        "--local-model",
        str(local_model),
    ]
    print(f"[esm-cache] {label}: computing missing embeddings {done}/{total}", flush=True)
    print("[esm-cache] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_esm_cache(cfg: TRIAGEConfig, args):
    if not args.auto_esm or not getattr(cfg, "use_esm", True):
        return
    pair_fasta = Path(cfg.pair_fourpack_dir) / "tuna.seq.fasta"
    if pair_fasta.exists():
        ensure_esm_cache_for_fasta(
            pair_fasta,
            Path(cfg.pair_esm_dir),
            args.esm_model,
            args.esm_batch,
            args.esm_max_len,
            "pair",
        )
    else:
        print(f"[esm-cache][warn] pair fasta missing: {pair_fasta}", flush=True)
    site_fasta = find_site_fasta(cfg)
    if site_fasta is None:
        site_cache_count = count_npy_files(Path(cfg.site_esm_dir))
        if site_cache_count > 0:
            print(f"[esm-cache] site: existing cache files={site_cache_count} dir={cfg.site_esm_dir}", flush=True)
        else:
            site_fasta = build_site_fasta_if_possible(cfg, args)
    if site_fasta is not None:
        ensure_esm_cache_for_fasta(
            site_fasta,
            Path(cfg.site_esm_dir),
            args.esm_model,
            args.esm_batch,
            args.esm_max_len,
            "site",
        )
    elif count_npy_files(Path(cfg.site_esm_dir)) > 0:
        pass
    else:
        print("[esm-cache][warn] site fasta not found; SITE will use PSSM/DSSP until a fasta is provided.", flush=True)


def ensure_pp_esm_cache(cfg: TRIAGEConfig, args):
    if not args.auto_esm or not getattr(cfg, "use_esm", True):
        return
    pp_root = args.pp_root or cfg.pp_root
    pp_fasta = Path(cfg.project_root) / "triage_pp_cache" / "dest_ppstyle.seq.fasta"
    lengths = parse_fasta_lengths(pp_fasta)
    if not lengths:
        built = build_pp_fasta(pp_root, pp_fasta)
        if built is None:
            print("[esm-cache][warn] PP fasta not available; GraphRBF-PP will use zero ESM fallback features.", flush=True)
            return
    ensure_esm_cache_for_fasta(
        pp_fasta,
        Path(cfg.pp_esm_dir),
        args.esm_model,
        args.esm_batch,
        args.esm_max_len,
        "dest_ppstyle",
    )


def ensure_dest_prepared(cfg: TRIAGEConfig) -> None:
    dest_root = Path(getattr(cfg, "dest_root", cfg.pp_root))
    raw_root = Path(getattr(cfg, "dest_raw_root", "data/Dset"))

    def nonempty_split(name: str) -> bool:
        path = dest_root / f"{name}.txt"
        return path.exists() and any(line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines())

    feature_dirs = [dest_root / x for x in ("labels", "pssm", "dssp", "seq")]
    feature_counts = [len(list(d.glob("*"))) if d.exists() else 0 for d in feature_dirs]
    if all(nonempty_split(x) for x in ("train", "val", "test")) and min(feature_counts) > 0:
        print(f"[dest-prepare] using prepared Dest root={dest_root} feature_counts={feature_counts}", flush=True)
        return
    script = Path(__file__).resolve().parent / "prepare_dest_from_graphrbf_pkl.py"
    if not script.exists():
        print(f"[dest-prepare][warn] missing script={script}; Dest split files may be incomplete.", flush=True)
        return
    cmd = [sys.executable, str(script), "--raw-root", str(raw_root), "--out-root", str(dest_root)]
    print(f"[dest-prepare] incomplete split files under {dest_root}; running: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def first_col(df: pd.DataFrame, names: List[str], default: Optional[str] = None) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def load_pssm_map(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = read_table(path)
    uid_col = first_col(df, ["uid", "chain_uid", "protein", "protein_id", "id"], df.columns[0])
    val_cols = [c for c in AA20 if c in df.columns]
    if len(val_cols) != 20:
        skip = {uid_col.lower(), "idx", "pos", "aa", "residue"}
        val_cols = [c for c in df.columns if c.lower() not in skip and pd.api.types.is_numeric_dtype(df[c])][:20]
    if len(val_cols) < 20:
        raise RuntimeError(f"{path} has no 20-column PSSM representation.")
    out = {}
    for uid, sub in df.groupby(uid_col, sort=False):
        out[str(uid)] = sub[val_cols[:20]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def load_esm_map(esm_dir: Path, max_len: int = 0) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not esm_dir or not Path(esm_dir).exists():
        return out
    for f in sorted(Path(esm_dir).glob("*.npy")):
        try:
            arr = np.load(f).astype(np.float32)
        except Exception as exc:
            print(f"[esm][warn] skip {f}: {exc}", flush=True)
            continue
        if arr.ndim != 2 or arr.shape[1] <= 0:
            print(f"[esm][warn] skip {f}: bad shape={arr.shape}", flush=True)
            continue
        if max_len > 0 and arr.shape[0] > max_len:
            arr = arr[:max_len]
        out[f.stem] = arr
        if "__" in f.stem:
            out.setdefault(f.stem.split("__", 1)[0], arr)
    print(f"[esm] loaded={len(out)} dir={esm_dir}", flush=True)
    return out


def load_dssp_map(path: Path, use_rsa: bool = True, use_ss: bool = True) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = read_table(path)
    uid_col = first_col(df, ["chain_uid", "uid", "protein", "protein_id", "id"], df.columns[0])
    ss_col = first_col(df, ["ss", "sec", "secondary_structure"])
    rsa_col = first_col(df, ["rsa", "rasa", "rel_asa"])
    out = {}
    for uid, sub in df.groupby(uid_col, sort=False):
        parts = []
        if use_ss:
            ss = sub[ss_col].astype(str).str.strip().str.upper().tolist() if ss_col else ["C"] * len(sub)
            arr = np.zeros((len(sub), len(SS8)), dtype=np.float32)
            for i, s in enumerate(ss):
                s = s if s in SS8 else "C"
                arr[i, SS8.index(s)] = 1.0
            parts.append(arr)
        if use_rsa:
            rsa = pd.to_numeric(sub[rsa_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if rsa_col else np.zeros(len(sub), dtype=np.float32)
            parts.append(rsa.reshape(-1, 1))
        if parts:
            out[str(uid)] = np.concatenate(parts, axis=1).astype(np.float32)
    return out


def make_residue_feature(
    uid: str,
    esm: Dict[str, np.ndarray],
    pssm: Dict[str, np.ndarray],
    dssp: Dict[str, np.ndarray],
    lengths: Dict[str, int],
    d_res_in: int,
) -> torch.Tensor:
    arrays = []
    if uid in esm:
        arrays.append(esm[uid])
    if uid in pssm:
        arrays.append(pssm[uid])
    if uid in dssp:
        arrays.append(dssp[uid])
    if arrays:
        L = max(a.shape[0] for a in arrays)
    else:
        L = int(lengths.get(uid, 1))
    fixed = []
    for a in arrays:
        if a.shape[0] < L:
            a = np.pad(a, ((0, L - a.shape[0]), (0, 0)))
        elif a.shape[0] > L:
            a = a[:L]
        fixed.append(a)
    feat = np.concatenate(fixed, axis=1) if fixed else np.zeros((L, 0), dtype=np.float32)
    if feat.shape[1] < d_res_in:
        feat = np.pad(feat, ((0, 0), (0, d_res_in - feat.shape[1])))
    elif feat.shape[1] > d_res_in:
        feat = feat[:, :d_res_in]
    return torch.from_numpy(feat.astype(np.float32))


def pad_residue_arrays(arrays: List[np.ndarray], d_res_in: int, length: Optional[int] = None) -> torch.Tensor:
    valid = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            continue
        valid.append(arr)
    if length is None:
        length = max([a.shape[0] for a in valid], default=1)
    fixed = []
    for arr in valid:
        if arr.shape[0] < length:
            arr = np.pad(arr, ((0, length - arr.shape[0]), (0, 0)))
        elif arr.shape[0] > length:
            arr = arr[:length]
        fixed.append(arr)
    feat = np.concatenate(fixed, axis=1) if fixed else np.zeros((length, 0), dtype=np.float32)
    if feat.shape[1] < d_res_in:
        feat = np.pad(feat, ((0, 0), (0, d_res_in - feat.shape[1])))
    elif feat.shape[1] > d_res_in:
        feat = feat[:, :d_res_in]
    return torch.from_numpy(feat.astype(np.float32))


def truncate_feature(feat: torch.Tensor, max_len: int) -> torch.Tensor:
    if max_len and max_len > 0 and feat.size(0) > max_len:
        return feat[:max_len].contiguous()
    return feat


class TriagePairDataset(Dataset):
    def __init__(self, cfg: TRIAGEConfig, max_items: int = 0):
        fourpack = Path(cfg.pair_fourpack_dir)
        shapes = read_table(fourpack / "tuna.shapes.tsv")
        self.a_col = first_col(shapes, ["A_uid", "protein_A", "uidA", "a", "protA"])
        self.b_col = first_col(shapes, ["B_uid", "protein_B", "uidB", "b", "protB"])
        self.pair_col = first_col(shapes, ["pair_id", "id", "pair"], shapes.columns[0])
        self.y_col = first_col(shapes, ["label", "y_pair", "has_contact", "target"])
        if not self.a_col or not self.b_col:
            raise RuntimeError(f"Cannot find pair columns in {fourpack / 'tuna.shapes.tsv'}: {list(shapes.columns)}")
        self.positive_pair_ids = set()
        self.contact_pair_labels: Dict[str, float] = {}
        contacts_path = fourpack / "tuna.contacts.tsv"
        if self.y_col is None and contacts_path.exists():
            contacts = read_table(contacts_path)
            c_pair = first_col(contacts, ["pair_id", "id", "pair"], contacts.columns[0])
            c_label = first_col(contacts, ["label", "y_pair", "has_contact", "target"])
            if c_label:
                lab = pd.to_numeric(contacts[c_label], errors="coerce").fillna(0.0)
                for pair_id, value in zip(contacts[c_pair].astype(str), lab.astype(float)):
                    self.contact_pair_labels[pair_id] = max(float(value), self.contact_pair_labels.get(pair_id, 0.0))
                self.positive_pair_ids = {pair_id for pair_id, value in self.contact_pair_labels.items() if value >= 0.5}
            else:
                self.positive_pair_ids = set(contacts[c_pair].astype(str).tolist())
        if max_items and max_items > 0:
            shapes = shapes.iloc[:max_items].copy()
        self.rows = shapes.reset_index(drop=True)
        self.lengths = parse_fasta_lengths(fourpack / "tuna.seq.fasta")
        self.esm = load_esm_map(Path(cfg.pair_esm_dir), max_len=cfg.max_pair_len) if getattr(cfg, "use_esm", True) else {}
        self.pssm = load_pssm_map(fourpack / "tuna.all.pssm.tsv") if cfg.use_pair_pssm else {}
        self.dssp = load_dssp_map(fourpack / "tuna.all.dssp.tsv", use_rsa=cfg.use_pair_dssp_rsa, use_ss=cfg.use_pair_dssp_ss)
        self.coords_dir = fourpack.parent / "coords"
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def _coords(self, uid: str, length: int) -> torch.Tensor:
        if length <= 0:
            return torch.empty(0, 3, dtype=torch.float32)
        for name in (uid, uid.lower(), uid.upper()):
            path = self.coords_dir / f"{name}.npy"
            if not path.exists():
                continue
            try:
                arr = np.load(path).astype(np.float32)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            arr = arr[:, :3]
            if arr.shape[0] >= length:
                return torch.from_numpy(arr[:length].copy())
            pad = np.full((length - arr.shape[0], 3), np.nan, dtype=np.float32)
            return torch.from_numpy(np.concatenate([arr, pad], axis=0))
        return torch.full((length, 3), float("nan"), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows.iloc[idx]
        a = str(r[self.a_col])
        b = str(r[self.b_col])
        pair_id = str(r[self.pair_col]) if self.pair_col else f"{a}__{b}"
        resA = truncate_feature(make_residue_feature(a, self.esm, self.pssm, self.dssp, self.lengths, self.cfg.d_res_in), self.cfg.max_pair_len)
        resB = truncate_feature(make_residue_feature(b, self.esm, self.pssm, self.dssp, self.lengths, self.cfg.d_res_in), self.cfg.max_pair_len)
        coordsA = self._coords(a, resA.size(0))
        coordsB = self._coords(b, resB.size(0))
        if self.y_col:
            y = float(r[self.y_col])
        elif self.contact_pair_labels:
            y = float(self.contact_pair_labels.get(pair_id, 0.0))
        else:
            y = float(pair_id in self.positive_pair_ids)
        return {
            "protein_A": a,
            "protein_B": b,
            "pair_id": pair_id,
            "resA": resA,
            "resB": resB,
            "coordsA": coordsA,
            "coordsB": coordsB,
            "maskA": torch.ones(resA.size(0)),
            "maskB": torch.ones(resB.size(0)),
            "y_pair": torch.tensor(y, dtype=torch.float32),
        }


class TriageSiteDataset(Dataset):
    def __init__(self, cfg: TRIAGEConfig, max_items: int = 0):
        site = Path(cfg.site_global_dir)
        contacts = read_table(site / "contacts.tsv")
        self.pair_col = first_col(contacts, ["pair_id", "complex", "id"], contacts.columns[0])
        self.a_col = first_col(contacts, ["A_chain", "chainA", "A_uid", "uidA"])
        self.b_col = first_col(contacts, ["B_chain", "chainB", "B_uid", "uidB"])
        self.i_col = first_col(contacts, ["i", "i_idx", "residue_A_index", "resA_idx", "idxA", "A_idx"])
        self.j_col = first_col(contacts, ["j", "j_idx", "residue_B_index", "resB_idx", "idxB", "B_idx"])
        if not all([self.pair_col, self.i_col, self.j_col]):
            raise RuntimeError(f"Cannot find site contact columns in {site / 'contacts.tsv'}: {list(contacts.columns)}")
        pairs = []
        for pair_id, sub in contacts.groupby(self.pair_col, sort=False):
            first = sub.iloc[0]
            pair_s = str(pair_id)
            a_chain = str(first[self.a_col]) if self.a_col else f"{pair_s}_A"
            b_chain = str(first[self.b_col]) if self.b_col else f"{pair_s}_B"
            pairs.append((pair_s, a_chain, b_chain, sub[[self.i_col, self.j_col]].copy()))
        if max_items and max_items > 0:
            pairs = pairs[:max_items]
        self.pairs = pairs
        self.esm = load_esm_map(Path(cfg.site_esm_dir), max_len=cfg.max_site_len) if getattr(cfg, "use_esm", True) else {}
        self.pssm = load_pssm_map(site / "pssm.all.tsv") if cfg.use_site_pssm else {}
        self.dssp = load_dssp_map(site / "dssp.all.tsv", use_rsa=cfg.use_site_dssp_rsa, use_ss=cfg.use_site_dssp_ss)
        self.lengths = {uid: arr.shape[0] for uid, arr in self.dssp.items()}
        for uid, arr in self.esm.items():
            self.lengths.setdefault(uid, arr.shape[0])
        for uid, arr in self.pssm.items():
            self.lengths.setdefault(uid, arr.shape[0])
        self.cfg = cfg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair_id, a, b, contacts = self.pairs[idx]
        resA = truncate_feature(make_residue_feature(a, self.esm, self.pssm, self.dssp, self.lengths, self.cfg.d_res_in), self.cfg.max_site_len)
        resB = truncate_feature(make_residue_feature(b, self.esm, self.pssm, self.dssp, self.lengths, self.cfg.d_res_in), self.cfg.max_site_len)
        y2d = torch.zeros(resA.size(0), resB.size(0), dtype=torch.float32)
        ii = pd.to_numeric(contacts[self.i_col], errors="coerce").dropna().astype(int).tolist()
        jj = pd.to_numeric(contacts[self.j_col], errors="coerce").dropna().astype(int).tolist()
        for i, j in zip(ii, jj):
            i0, j0 = int(i) - 1, int(j) - 1
            if 0 <= i0 < y2d.size(0) and 0 <= j0 < y2d.size(1):
                y2d[i0, j0] = 1.0
        return {
            "complex": pair_id,
            "protein_A": a,
            "protein_B": b,
            "resA": resA,
            "resB": resB,
            "maskA": torch.ones(resA.size(0)),
            "maskB": torch.ones(resB.size(0)),
            "y2d": y2d,
            "y_res_A": y2d.max(dim=1).values,
            "y_res_B": y2d.max(dim=0).values,
            "y_pair": torch.tensor(float(y2d.sum() > 0), dtype=torch.float32),
        }


def read_id_list(path: Path) -> List[str]:
    def clean_id(text: str) -> str:
        return re.sub(r"\.(npy|npz|pt|fa|fasta|faa|txt|tsv)(\.gz)?$", "", text.strip().split()[0], flags=re.I)

    if not path.exists():
        return []
    out = []
    seen = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            pid = clean_id(line)
            if pid and pid not in seen:
                out.append(pid)
                seen.add(pid)
    return out


def l1_sequence_for_id(root: Path, pid: str, target_len: int = 0) -> str:
    for seq_dir_name in ("seq", "SEQ", "fasta"):
        seq_dir = root / seq_dir_name
        if not seq_dir.exists():
            continue
        seqs = []
        for ext in (".fa", ".fasta", ".faa", ".txt"):
            seq = read_fasta_sequence(seq_dir / f"{pid}{ext}")
            if seq:
                seqs.append(seq)
        if target_len > 0:
            for seq in seqs:
                if len(seq) == target_len:
                    return seq
        if seqs:
            return seqs[0]
    return ""


def find_l1_feature_file(root: Path, subdirs: Tuple[str, ...], pid: str) -> Optional[Path]:
    names = [pid, pid.lower(), pid.upper()]
    suffixes = (".npy", ".npz", ".npz.gz", ".pt")
    for subdir in subdirs:
        d = root / subdir
        if not d.exists():
            continue
        for name in names:
            for suffix in suffixes:
                path = d / f"{name}{suffix}"
                if path.exists():
                    return path
    for name in names:
        protein_dir = root / name
        if not protein_dir.exists():
            continue
        for subdir in subdirs:
            d = protein_dir / subdir
            if d.exists():
                for suffix in suffixes:
                    hits = sorted(d.glob(f"*{suffix}"))
                    if hits:
                        return hits[0]
            for suffix in suffixes:
                path = protein_dir / f"{subdir}{suffix}"
                if path.exists():
                    return path
    return None


def valid_l1_label_ids(root: Path, ids: List[str]) -> List[str]:
    out = []
    for pid in ids:
        label_path = find_l1_feature_file(root, ("labels", "label", "annotations", "annotations_case_realdata"), pid)
        if label_path is None:
            continue
        try:
            y = load_l1_array(label_path, "labels").reshape(-1)
        except Exception:
            continue
        if y.size <= 0:
            continue
        seq = l1_sequence_for_id(root, pid, int(y.shape[0]))
        if seq and len(seq) != int(y.shape[0]):
            continue
        out.append(pid)
    return out


def l1_label_pos_count(root: Path, pid: str) -> Optional[int]:
    label_path = find_l1_feature_file(root, ("labels", "label", "annotations", "annotations_case_realdata"), pid)
    if label_path is None:
        return None
    try:
        y = load_l1_array(label_path, "labels").reshape(-1)
    except Exception:
        return None
    if y.size <= 0:
        return None
    return int((y > 0.5).sum())


def filter_l1_zero_label_ids(root: Path, ids: List[str]) -> Tuple[List[str], int]:
    kept: List[str] = []
    removed = 0
    for pid in ids:
        pos = l1_label_pos_count(root, pid)
        if pos is not None and pos <= 0:
            removed += 1
            continue
        kept.append(pid)
    return kept, removed


def l1_label_bucket(root: Path, pid: str) -> str:
    label_path = find_l1_feature_file(root, ("labels", "label", "annotations", "annotations_case_realdata"), pid)
    if label_path is None:
        return "unknown"
    try:
        y = load_l1_array(label_path, "labels").reshape(-1)
    except Exception:
        return "unknown"
    if y.size <= 0:
        return "unknown"
    pos = float((y > 0.5).sum())
    total = float(y.size)
    frac = pos / max(1.0, total)
    if pos <= 0:
        return "zero"
    if pos >= total:
        return "full"
    if frac < 0.10:
        return "low"
    if frac < 0.50:
        return "mid"
    if frac < 0.90:
        return "high"
    return "very_high"


def split_l1_ids_8_1_1(root: Path, ids: List[str], split: str, seed: int = 42) -> List[str]:
    def split_once(cur_seed: int) -> Tuple[List[str], List[str], List[str]]:
        rng = random.Random(cur_seed)
        groups: Dict[str, List[str]] = defaultdict(list)
        for pid in ids:
            groups[l1_label_bucket(root, pid)].append(pid)
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        for bucket in sorted(groups):
            group = list(groups[bucket])
            rng.shuffle(group)
            n = len(group)
            n_train = int(round(n * 0.80))
            n_val = int(round(n * 0.10))
            n_train = max(0, min(n_train, n))
            n_val = max(0, min(n_val, n - n_train))
            if n >= 3 and n_val == 0:
                n_val = 1
                if n_train + n_val > n:
                    n_train = n - n_val
            if n >= 3 and n - n_train - n_val == 0:
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(0, n_val - 1)
            train_ids.extend(group[:n_train])
            val_ids.extend(group[n_train:n_train + n_val])
            test_ids.extend(group[n_train + n_val:])
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)
        return train_ids, val_ids, test_ids

    def pos_rate(part: List[str]) -> float:
        pos = 0.0
        total = 0.0
        for pid in part:
            label_path = find_l1_feature_file(root, ("labels", "label", "annotations", "annotations_case_realdata"), pid)
            if label_path is None:
                continue
            try:
                y = load_l1_array(label_path, "labels").reshape(-1)
            except Exception:
                continue
            pos += float((y > 0.5).sum())
            total += float(y.size)
        return pos / max(1.0, total)

    def bucket_counts(part: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for pid in part:
            out[l1_label_bucket(root, pid)] += 1
        return out

    def balance_score(parts: Tuple[List[str], List[str], List[str]]) -> float:
        all_part = list(parts[0]) + list(parts[1]) + list(parts[2])
        all_rate = pos_rate(all_part)
        score = sum(abs(pos_rate(part) - all_rate) for part in parts)
        all_b = bucket_counts(all_part)
        n_all = max(1, len(all_part))
        for part in parts:
            bc = bucket_counts(part)
            n = max(1, len(part))
            for bucket, count in all_b.items():
                score += 0.15 * abs((bc.get(bucket, 0) / n) - (count / n_all))
        return score

    best = None
    best_score = float("inf")
    for i in range(256):
        cand = split_once(seed + i)
        score = balance_score(cand)
        if score < best_score:
            best = cand
            best_score = score
    train_ids, val_ids, test_ids = best if best is not None else split_once(seed)
    if split == "train":
        return train_ids
    if split == "val":
        return val_ids
    return test_ids


def l1_label_balance(root: Path, ids: List[str], cap: float = 12.0) -> Dict[str, float]:
    pos = 0.0
    neg = 0.0
    proteins = 0
    zero_label = 0
    full_label = 0
    buckets: Dict[str, int] = defaultdict(int)
    for pid in ids:
        label_path = find_l1_feature_file(root, ("labels", "label", "annotations", "annotations_case_realdata"), pid)
        if label_path is None:
            continue
        try:
            y = load_l1_array(label_path, "labels").reshape(-1)
        except Exception:
            continue
        if y.size <= 0:
            continue
        yp = float((y > 0.5).sum())
        yn = float(y.size) - yp
        pos += yp
        neg += yn
        proteins += 1
        if yp <= 0:
            zero_label += 1
        elif yn <= 0:
            full_label += 1
        buckets[l1_label_bucket(root, pid)] += 1
    raw_pos_weight = neg / max(1.0, pos)
    eff_pos_weight = max(1.0, min(float(cap), raw_pos_weight))
    weighted_pos = pos * eff_pos_weight
    weighted_pos_frac = weighted_pos / max(1.0, weighted_pos + neg)
    out = {
        "proteins": float(proteins),
        "pos": pos,
        "neg": neg,
        "pos_rate": pos / max(1.0, pos + neg),
        "raw_pos_weight": raw_pos_weight,
        "eff_pos_weight": eff_pos_weight,
        "weighted_pos_frac": weighted_pos_frac,
        "zero_label": float(zero_label),
        "full_label": float(full_label),
    }
    for key, value in buckets.items():
        out[f"bucket_{key}"] = float(value)
    return out


def print_l1_label_balance(dataset_label: str, root: Path, split: str, ids: List[str], cap: float):
    stats = l1_label_balance(root, ids, cap=cap)
    bucket_text = " ".join(
        f"{k.replace('bucket_', '')}={int(v)}"
        for k, v in sorted(stats.items())
        if k.startswith("bucket_")
    )
    print(
        f"[label-balance][{dataset_label}] {split} proteins={int(stats['proteins'])} "
        f"res_pos={int(stats['pos'])} res_neg={int(stats['neg'])} "
        f"pos_rate={stats['pos_rate']:.4f} raw_pos_weight={stats['raw_pos_weight']:.2f} "
        f"eff_pos_weight={stats['eff_pos_weight']:.2f} weighted_pos_frac={stats['weighted_pos_frac']:.3f} "
        f"zero_label={int(stats['zero_label'])} full_label={int(stats['full_label'])} "
        f"buckets=({bucket_text})",
        flush=True,
    )


def print_l1_structure_coverage(dataset_label: str, root: Path, split: str, ids: List[str]):
    n_coord = 0
    n_pdb = 0
    for pid in ids:
        if find_l1_feature_file(root, ("coords", "coordinates"), pid) is not None:
            n_coord += 1
        elif find_l1_pdb_file(root, pid) is not None:
            n_pdb += 1
    total = max(1, len(ids))
    print(
        f"[structure][{dataset_label}] {split} coords_np={n_coord}/{len(ids)} "
        f"pdb={n_pdb}/{len(ids)} any={(n_coord + n_pdb)}/{len(ids)} "
        f"coverage={(n_coord + n_pdb) / total:.3f}",
        flush=True,
    )


def read_l1_split_ids(root: Path, split: str, seed: int = 42, id_list_path: Optional[str] = None) -> List[str]:
    split_path = root / f"{split}.txt"
    if split_path.exists():
        ids = read_id_list(split_path)
        if ids:
            return ids
        print(f"[split-warn] empty split file ignored: {split_path}", flush=True)
    legacy_split = root.parent / f"{root.name}_split_{split}.txt"
    if legacy_split.exists():
        ids = read_id_list(legacy_split)
        if ids:
            return ids
        print(f"[split-warn] empty legacy split file ignored: {legacy_split}", flush=True)
    all_ids_path = root / "all_ids.txt"
    if all_ids_path.exists():
        ids = read_id_list(all_ids_path)
    elif id_list_path and Path(id_list_path).exists():
        ids = valid_l1_label_ids(root, read_id_list(Path(id_list_path)))
    else:
        ids = []
        for subdir in ("labels", "label", "annotations", "annotations_case_realdata"):
            d = root / subdir
            if d.exists():
                ids.extend([p.stem for p in sorted(d.glob("*.npy"))])
        ids = sorted(set(ids))
    if not ids:
        return []
    return split_l1_ids_8_1_1(root, ids, split, seed)


def find_l1_npy(root: Path, subdirs: Tuple[str, ...], pid: str) -> Optional[Path]:
    names = [pid, pid.lower(), pid.upper()]
    for subdir in subdirs:
        d = root / subdir
        if not d.exists():
            continue
        for name in names:
            path = d / f"{name}.npy"
            if path.exists():
                return path
    for name in names:
        protein_dir = root / name
        if not protein_dir.exists():
            continue
        for subdir in subdirs:
            d = protein_dir / subdir
            if d.exists():
                hits = sorted(d.glob("*.npy"))
                if hits:
                    return hits[0]
            path = protein_dir / f"{subdir}.npy"
            if path.exists():
                return path
    return None


def load_l1_array(path: Path, kind: str = "") -> np.ndarray:
    if path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy().astype(np.float32)
        return np.asarray(obj, dtype=np.float32)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            loaded = np.load(f, allow_pickle=True)
            return _loaded_l1_array_to_numpy(loaded, kind, path)
    loaded = np.load(path, allow_pickle=True)
    return _loaded_l1_array_to_numpy(loaded, kind, path)


def _loaded_l1_array_to_numpy(loaded, kind: str, path: Path) -> np.ndarray:
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys_by_kind = {
                "labels": ("labels", "label", "y", "data"),
                "pssm": ("pssm", "PSSM", "data"),
                "dssp": ("dssp", "DSSP", "ss", "data"),
                "coords": ("ca", "CA", "coords", "xyz", "data"),
            }
            for key in keys_by_kind.get(kind, ("data",)):
                if key in loaded.files:
                    return loaded[key].astype(np.float32)
            for key in loaded.files:
                arr = np.asarray(loaded[key])
                if arr.dtype != object:
                    return arr.astype(np.float32)
        finally:
            loaded.close()
        raise ValueError(f"No numeric array found in {path}")
    return np.asarray(loaded, dtype=np.float32)


def load_pdb_ca_coords(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    coords = []
    seen = set()
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    try:
        with opener(path, mode, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                key = (line[21:22].strip(), line[22:27].strip(), line[26:27].strip())
                if key in seen:
                    continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    seen.add(key)
                except Exception:
                    continue
    except Exception:
        return None
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def find_l1_pdb_file(root: Path, pid: str) -> Optional[Path]:
    struct_dirs = ("structures", "structure", "structures_af", "pdb")
    explicit_names = []
    for name in (pid, pid.lower(), pid.upper()):
        explicit_names.extend([f"{name}.pdb", f"{name}.ent", f"{name}.pdb.gz", f"{name}.ent.gz"])
    for subdir in struct_dirs:
        d = root / subdir
        if not d.exists():
            continue
        for name in explicit_names:
            path = d / name
            if path.exists():
                return path
        for pattern in (f"AF-{pid}-F1-*.pdb", f"AF-{pid}-F1-*.ent", f"{pid}*.pdb", f"{pid}*.ent"):
            hits = sorted(d.glob(pattern))
            if hits:
                return hits[0]
    return None


def load_l1_coords(root: Path, pid: str) -> Optional[np.ndarray]:
    path = find_l1_feature_file(root, ("coords", "coordinates"), pid)
    if path is not None:
        arr = load_l1_array(path, "coords").astype(np.float32)
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            return arr.reshape(-1, arr.shape[-1])[:, :3]
    pdb_path = find_l1_pdb_file(root, pid)
    if pdb_path is not None:
        return load_pdb_ca_coords(pdb_path)
    return None


class GraphRBFPPDataset(Dataset):
    def __init__(
        self,
        cfg: TRIAGEConfig,
        pp_root: str,
        split: str,
        max_items: int = 0,
        esm_dir: Optional[str] = None,
        id_list_path: Optional[str] = None,
        dataset_label: str = "GraphRBF-PP",
    ):
        root = Path(pp_root)
        split_path = root / f"{split}.txt"
        self.root = root
        self.cfg = cfg
        self.split = split
        self.dataset_label = dataset_label
        pids = read_l1_split_ids(root, split, id_list_path=id_list_path)
        if not pids:
            raise FileNotFoundError(
                f"{dataset_label} split has no items for {split}: expected {split_path}, "
                f"{root / 'all_ids.txt'}, an accession list, or label .npy files under {root}"
            )
        if bool(getattr(cfg, "l1_exclude_zero_label_proteins", False)):
            pids, removed_zero = filter_l1_zero_label_ids(root, pids)
            if removed_zero > 0:
                print(
                    f"[label-filter] {dataset_label} {split} excluded_zero_label={removed_zero} kept={len(pids)}",
                    flush=True,
                )
            if not pids:
                raise FileNotFoundError(
                    f"{dataset_label} split has no non-zero-label items for {split}; "
                    f"run repair_rbp400_zero_labels.py or disable cfg.l1_exclude_zero_label_proteins"
                )
        legacy_split = root.parent / f"{root.name}_split_{split}.txt"
        if split_path.exists():
            print(
                f"[split-fixed] {dataset_label} {split}=file path={split_path}",
                flush=True,
            )
        elif legacy_split.exists():
            print(
                f"[split-fixed] {dataset_label} {split}=legacy path={legacy_split}",
                flush=True,
            )
        else:
            print(
                f"[split-auto] {dataset_label} {split}=auto stratified_label_search 8:1:1 seed=42 trials=256 root={root}",
                flush=True,
            )
        if max_items and max_items > 0:
            pids = pids[:max_items]
        self.pids = pids
        self.manifest = {}
        manifest_path = root / "pp_manifest.tsv"
        if manifest_path.exists():
            df = read_table(manifest_path)
            if "pid" in df.columns:
                self.manifest = {str(r["pid"]): r for _, r in df.iterrows()}
        esm_root = esm_dir if esm_dir is not None else cfg.pp_esm_dir
        self.esm = load_esm_map(Path(esm_root), max_len=cfg.max_site_len) if getattr(cfg, "use_esm", True) else {}

    def __len__(self):
        return len(self.pids)

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

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx]
        y = self._npy("labels", pid)
        if y is None:
            raise FileNotFoundError(f"{self.dataset_label} label missing for {pid} under {self.root}")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        L = int(y.shape[0])
        pssm = self._npy("pssm", pid)
        dssp = self._npy("dssp", pid)
        coords = load_l1_coords(self.root, pid)
        esm_arr = self.esm.get(pid)
        esm_fallback = np.zeros((L, 1280), dtype=np.float32) if getattr(self.cfg, "use_esm", True) and esm_arr is None else None
        resA = truncate_feature(pad_residue_arrays([esm_arr, esm_fallback, pssm, dssp], self.cfg.d_res_in, length=L), self.cfg.max_site_len)
        y_t = torch.from_numpy(y[: resA.size(0)].astype(np.float32))
        coords_t = None
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if coords.ndim == 2 and coords.shape[1] >= 3:
                coord_arr = np.full((resA.size(0), 3), np.nan, dtype=np.float32)
                n_coord = min(int(coords.shape[0]), int(resA.size(0)))
                if n_coord > 0:
                    coord_arr[:n_coord, :3] = coords[:n_coord, :3].astype(np.float32)
                coords_t = torch.from_numpy(coord_arr)
        dummyB = torch.zeros(1, self.cfg.d_res_in, dtype=torch.float32)
        return {
            "protein_A": pid,
            "protein_B": f"{pid}__dummy",
            "resA": resA,
            "resB": dummyB,
            "maskA": torch.ones(resA.size(0)),
            "maskB": torch.ones(1),
            "coordsA": coords_t if coords_t is not None else torch.full((resA.size(0), 3), float("nan")),
            "coordsB": torch.zeros(1, 3),
            "y_res_A": y_t,
        }


class DestTriagePairDataset(Dataset):
    def __init__(
        self,
        cfg: TRIAGEConfig,
        root: str,
        split: str,
        max_items: int = 0,
        esm_dir: Optional[str] = None,
        dataset_label: str = "Dest-triage",
    ):
        self.root = Path(root)
        self.cfg = cfg
        self.split = split
        self.dataset_label = dataset_label
        split_ids = set(read_l1_split_ids(self.root, split))
        manifest_path = self.root / "dest_manifest.tsv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{dataset_label} requires {manifest_path}")
        manifest = read_table(manifest_path)
        if "pid" not in manifest.columns or "pdb_id" not in manifest.columns:
            raise RuntimeError(f"{manifest_path} must contain pid and pdb_id columns")
        rows = []
        all_by_pdb: Dict[str, List[str]] = defaultdict(list)
        pid_to_pdb: Dict[str, str] = {}
        for _, r in manifest.iterrows():
            pid_all = str(r["pid"])
            pdb_all = str(r["pdb_id"])
            all_by_pdb[pdb_all].append(pid_all)
            pid_to_pdb[pid_all] = pdb_all
        for _, r in manifest.iterrows():
            pid = str(r["pid"])
            if pid in split_ids:
                rows.append(r)
        by_pdb: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            by_pdb[str(r["pdb_id"])].append(str(r["pid"]))
        pairs: List[Tuple[str, str, str]] = []
        mode = str(getattr(cfg, "dest_pairing_mode", "same_pdb_or_self")).lower()
        max_partners = max(1, int(getattr(cfg, "dest_pair_max_partners", 1)))
        split_pid_set = set(split_ids)
        if mode in ("same_pdb_or_self", "anchor_all", "self_fallback"):
            for r in rows:
                a = str(r["pid"])
                pdb_id = str(r["pdb_id"])
                candidates = [p for p in sorted(set(by_pdb.get(pdb_id, []))) if p != a]
                if not candidates:
                    candidates = [a]
                for b in candidates[:max_partners]:
                    tag = "self" if a == b else "samepdb"
                    pairs.append((f"{pdb_id}:{a}__{b}:{tag}", a, b))
        elif mode == "same_pdb_all":
            for pdb_id, pids in by_pdb.items():
                pids = sorted(set(pids))
                if len(pids) < 2:
                    continue
                for a, b in itertools.combinations(pids, 2):
                    pairs.append((f"{pdb_id}:{a}__{b}", a, b))
        else:
            raise ValueError(f"Unknown Dest pairing mode: {mode}")
        if max_items and max_items > 0:
            pairs = pairs[:max_items]
        if not pairs:
            raise FileNotFoundError(f"{dataset_label} split={split} has no same-PDB chain pairs under {self.root}")
        self.pairs = pairs
        n_self = sum(1 for _, a, b in pairs if a == b)
        print(
            f"[dest-pairs] {dataset_label} split={split} mode={mode} pairs={len(pairs)} "
            f"self_fallback={n_self} same_pdb={len(pairs) - n_self}",
            flush=True,
        )
        esm_root = esm_dir if esm_dir is not None else getattr(cfg, "dest_esm_dir", cfg.pp_esm_dir)
        self.esm = load_esm_map(Path(esm_root), max_len=cfg.max_site_len) if getattr(cfg, "use_esm", True) else {}

    def __len__(self):
        return len(self.pairs)

    def _npy(self, subdir: str, pid: str) -> Optional[np.ndarray]:
        aliases = {
            "labels": ("labels", "label", "annotations", "annotations_case_realdata"),
            "pssm": ("pssm", "PSSM"),
            "dssp": ("dssp", "DSSP"),
            "coords": ("coords", "coordinates"),
        }
        path = find_l1_feature_file(self.root, aliases.get(subdir, (subdir,)), pid)
        if path is None:
            return None
        return load_l1_array(path, subdir).astype(np.float32)

    def _chain(self, pid: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self._npy("labels", pid)
        if y is None:
            raise FileNotFoundError(f"{self.dataset_label} label missing for {pid} under {self.root}")
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        L = int(y.shape[0])
        pssm = self._npy("pssm", pid)
        dssp = self._npy("dssp", pid)
        esm_arr = self.esm.get(pid)
        esm_fallback = np.zeros((L, 1280), dtype=np.float32) if getattr(self.cfg, "use_esm", True) and esm_arr is None else None
        res = truncate_feature(pad_residue_arrays([esm_arr, esm_fallback, pssm, dssp], self.cfg.d_res_in, length=L), self.cfg.max_site_len)
        y_t = torch.from_numpy(y[: res.size(0)].astype(np.float32))
        coords = load_l1_coords(self.root, pid)
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            coord_arr = np.full((res.size(0), 3), np.nan, dtype=np.float32)
            if coords.ndim == 2 and coords.shape[1] >= 3:
                n_coord = min(int(coords.shape[0]), int(res.size(0)))
                coord_arr[:n_coord, :3] = coords[:n_coord, :3].astype(np.float32)
            coords_t = torch.from_numpy(coord_arr)
        else:
            coords_t = torch.full((res.size(0), 3), float("nan"))
        return res, y_t, coords_t

    def __getitem__(self, idx: int) -> Dict:
        pair_id, a, b = self.pairs[idx]
        resA, yA, coordsA = self._chain(a)
        resB, yB, coordsB = self._chain(b)
        y2d = torch.outer(yA.float(), yB.float())
        return {
            "complex": pair_id,
            "protein_A": a,
            "protein_B": b,
            "resA": resA,
            "resB": resB,
            "maskA": torch.ones(resA.size(0)),
            "maskB": torch.ones(resB.size(0)),
            "coordsA": coordsA,
            "coordsB": coordsB,
            "y_res_A": yA,
            "y_res_B": yB,
            "y2d": y2d,
            "y_pair": torch.tensor(float((yA.sum() > 0) and (yB.sum() > 0)), dtype=torch.float32),
        }


def collate_pad(batch):
    out = {}
    tensor_keys = [k for k, v in batch[0].items() if torch.is_tensor(v)]
    for k in batch[0]:
        if k not in tensor_keys:
            out[k] = [b[k] for b in batch]
    for k in tensor_keys:
        if k == "y_pair":
            out[k] = torch.stack([b[k] for b in batch]).view(-1)
        elif k == "y2d":
            la = max(b[k].shape[0] for b in batch)
            lb = max(b[k].shape[1] for b in batch)
            out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, lb - b[k].shape[1], 0, la - b[k].shape[0])) for b in batch])
        elif batch[0][k].dim() == 1:
            L = max(b[k].shape[0] for b in batch)
            out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, L - b[k].shape[0])) for b in batch])
        elif batch[0][k].dim() == 2:
            L = max(b[k].shape[0] for b in batch)
            out[k] = torch.stack([torch.nn.functional.pad(b[k], (0, 0, 0, L - b[k].shape[0])) for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_batch(model: TRIAGEPPIModel, batch: Dict) -> Dict:
    return model(
        resA=batch["resA"],
        maskA=batch.get("maskA"),
        chainA=batch.get("chainA"),
        coordsA=batch.get("coordsA"),
        resB=batch["resB"],
        maskB=batch.get("maskB"),
        chainB=batch.get("chainB"),
        coordsB=batch.get("coordsB"),
    )


def binary_metrics(prob: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    prob = prob.detach().float().cpu().view(-1)
    y = y.detach().float().cpu().view(-1)
    pred = (prob >= thr).float()
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    denom = math.sqrt(max(1e-8, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    order = torch.argsort(prob, descending=True)
    yy = y[order]
    if yy.sum() > 0:
        precision_curve = torch.cumsum(yy, 0) / torch.arange(1, yy.numel() + 1, dtype=torch.float32)
        auprc = float((precision_curve * yy).sum() / yy.sum())
    else:
        auprc = 0.0
    n_pos = float(y.sum())
    n_neg = float(y.numel() - y.sum())
    if n_pos > 0 and n_neg > 0:
        tp_curve = torch.cumsum(yy, 0)
        fp_curve = torch.cumsum(1.0 - yy, 0)
        tpr = tp_curve / max(n_pos, 1.0)
        fpr = fp_curve / max(n_neg, 1.0)
        tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
        fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
        auroc = float(torch.trapz(tpr, fpr))
    else:
        auroc = float("nan")
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc, "auprc": auprc, "auroc": auroc}


def _ceil_fraction_len(length: int, denom: int) -> int:
    return max(1, int(math.ceil(max(1, int(length)) / float(max(1, int(denom))))))


def residue_topk_metrics(prob: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    prob = prob.detach().float().cpu()
    y = y.detach().float().cpu()
    mask = mask.detach().cpu().bool()
    sums = {
        "l1_recall_l5": 0.0,
        "l1_recall_l10": 0.0,
        "l1_precision_10": 0.0,
        "l1_hit_2": 0.0,
        "l1_hit_20": 0.0,
    }
    n = 0
    for b in range(prob.size(0)):
        valid = mask[b]
        if valid.sum().item() <= 0:
            continue
        p = prob[b][valid]
        lab = (y[b][valid] > 0.5).float()
        positives = float(lab.sum().item())
        if positives <= 0:
            continue
        L = int(p.numel())

        def hits_at(k: int) -> float:
            kk = max(1, min(int(k), L))
            idx = torch.topk(p, kk, largest=True).indices
            return float(lab[idx].sum().item())

        h_l5 = hits_at(_ceil_fraction_len(L, 5))
        h_l10 = hits_at(_ceil_fraction_len(L, 10))
        h_10 = hits_at(10)
        h_2 = hits_at(2)
        h_20 = hits_at(20)
        sums["l1_recall_l5"] += h_l5 / max(1.0, positives)
        sums["l1_recall_l10"] += h_l10 / max(1.0, positives)
        sums["l1_precision_10"] += h_10 / float(max(1, min(10, L)))
        sums["l1_hit_2"] += 1.0 if h_2 > 0 else 0.0
        sums["l1_hit_20"] += 1.0 if h_20 > 0 else 0.0
        n += 1
    if n <= 0:
        return {**sums, "l1_topk_n": 0.0}
    out = {k: v / n for k, v in sums.items()}
    out["l1_topk_n"] = float(n)
    return out


def apply_l1_ager(prob: torch.Tensor, coords: Optional[torch.Tensor], mask: torch.Tensor, cfg: TRIAGEConfig) -> torch.Tensor:
    if not bool(getattr(cfg, "l1_ager_enable", False)) or coords is None:
        return prob
    p = prob.detach().float()
    c = coords.detach().float().to(device=p.device)
    m = mask.detach().to(device=p.device).bool()
    if c.dim() != 3 or c.shape[:2] != p.shape:
        return prob
    finite = torch.isfinite(c).all(dim=-1) & m
    if not bool(finite.any()):
        return prob
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    radius = float(getattr(cfg, "l1_ager_radius", 10.0))
    alpha = float(getattr(cfg, "l1_ager_alpha", 0.30))
    top_m = max(1, int(getattr(cfg, "l1_ager_top_m", 5)))
    smoothed = p.clone()
    for b in range(p.size(0)):
        valid = finite[b]
        if valid.sum().item() <= 1:
            continue
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        cc = c[b, idx]
        pp = p[b, idx]
        dist = torch.cdist(cc, cc)
        local_vals = []
        for i in range(idx.numel()):
            neigh = (dist[i] <= radius)
            vals = pp[neigh]
            if vals.numel() > top_m:
                vals = torch.topk(vals, top_m, largest=True).values
            local_vals.append(vals.mean())
        local = torch.stack(local_vals)
        smoothed[b, idx] = (pp + alpha * (local - pp).clamp_min(0.0)).clamp(0.0, 1.0)
    return smoothed.to(dtype=prob.dtype, device=prob.device)


def fbeta_score(precision: float, recall: float, beta: float) -> float:
    b2 = float(beta) * float(beta)
    return (1.0 + b2) * precision * recall / max(1e-8, b2 * precision + recall)


def binary_metrics_best_threshold(
    prob: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1.0,
    objective_weights: Optional[Dict[str, float]] = None,
    balance_w: float = 0.0,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    mode: str = "auto_mcc",
) -> Dict[str, float]:
    prob = prob.detach().float().cpu().view(-1)
    y = y.detach().float().cpu().view(-1)
    valid = torch.isfinite(prob) & torch.isfinite(y)
    prob, y = prob[valid], y[valid]
    if y.numel() == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "auprc": 0.0, "auroc": float("nan"), "thr": 0.5}
    base = binary_metrics(prob, y, thr=0.5)
    uniq = torch.unique(prob)
    if uniq.numel() > 512:
        qs = torch.linspace(0.01, 0.99, 199)
        thresholds = torch.quantile(prob, qs).unique()
    else:
        thresholds = uniq
    best = dict(base)
    best["thr"] = 0.5
    best["fbeta"] = fbeta_score(best["precision"], best["recall"], beta)
    objective_weights = objective_weights or {}
    mode = str(mode or "auto_mcc").lower()

    def objective(m: Dict[str, float]) -> float:
        if mode in ("auto_acc", "acc", "val_acc"):
            return m["acc"]
        if mode in ("auto_recall_floor", "recall_floor", "val_recall_floor"):
            if m["recall"] + 1e-12 < float(min_recall):
                return -1e9
            return max(0.0, m["mcc"]) + 0.25 * m["f1"]
        if mode in ("fixed_0.5", "fixed05", "threshold_0.5"):
            return 1.0 if abs(float(m["thr"]) - 0.5) < 1e-12 else -1e9
        if mode in ("auto_f1", "f1", "val_f1"):
            return m["f1"]
        if mode in ("auto_fbeta", "fbeta", "val_fbeta"):
            return m["fbeta"]
        if mode in ("auto_posrate", "posrate", "prevalence"):
            return -abs(float((prob >= float(m["thr"])).float().mean()) - float(y.mean())) + 1e-3 * max(0.0, m["mcc"])
        weighted = 0.0
        active = False
        for key in ("acc", "precision", "recall", "f1", "mcc"):
            w = float(objective_weights.get(key, 0.0))
            if w:
                active = True
                weighted += w * (max(0.0, m[key]) if key == "mcc" else m[key])
        score = weighted if active else (m["fbeta"] if beta > 1.0 else m["mcc"] + m["f1"])
        if balance_w > 0:
            score += float(balance_w) * min(m["precision"], m["recall"], m["f1"], max(0.0, m["mcc"]))
        if min_precision > 0:
            score -= max(0.0, float(min_precision) - m["precision"]) * 0.50
        if min_recall > 0:
            score -= max(0.0, float(min_recall) - m["recall"]) * 0.50
        return score

    fallback = dict(best)
    fallback_key = (fallback["recall"], fallback["mcc"], fallback["f1"])
    best_obj = objective(best)
    best_key = (best_obj, best["mcc"], best["f1"], best["recall"])
    for t in thresholds:
        m = binary_metrics(prob, y, thr=float(t))
        m["thr"] = float(t)
        m["fbeta"] = fbeta_score(m["precision"], m["recall"], beta)
        fb_key = (m["recall"], m["mcc"], m["f1"])
        if fb_key > fallback_key:
            fallback = dict(m)
            fallback_key = fb_key
        key = (objective(m), m["mcc"], m["f1"], m["recall"])
        if key > best_key:
            best = dict(m)
            best["thr"] = float(t)
            best_key = key
    if mode in ("auto_recall_floor", "recall_floor", "val_recall_floor") and best_key[0] <= -1e8:
        return fallback
    return best


def auprc_score(prob: torch.Tensor, y: torch.Tensor) -> float:
    prob = prob.detach().float().cpu().view(-1)
    y = y.detach().float().cpu().view(-1)
    valid = torch.isfinite(prob) & torch.isfinite(y)
    prob, y = prob[valid], y[valid]
    if y.numel() == 0 or y.sum() <= 0:
        return 0.0
    order = torch.argsort(prob, descending=True)
    yy = y[order]
    precision_curve = torch.cumsum(yy, 0) / torch.arange(1, yy.numel() + 1, dtype=torch.float32)
    return float((precision_curve * yy).sum() / yy.sum())


def l2_precision_at_topm(out: Dict, batch: Dict) -> float:
    if "y2d" not in batch:
        return 0.0
    idx = out["topm_interface_idx"].detach().cpu()
    y2d = batch["y2d"].detach().cpu()
    vals = []
    for b in range(idx.size(0)):
        vals.append(y2d[b, idx[b, :, 0], idx[b, :, 1]].float().mean())
    return float(torch.stack(vals).mean()) if vals else 0.0


def l2_topl_precision(out: Dict, batch: Dict) -> float:
    if "y2d" not in batch:
        return 0.0
    scores = out["S_interface"].detach().cpu()
    y2d = batch["y2d"].detach().cpu()
    maskA = batch.get("maskA", torch.ones(scores.shape[:2])).detach().cpu() > 0.5
    maskB = batch.get("maskB", torch.ones(scores.size(0), scores.size(2))).detach().cpu() > 0.5
    vals = []
    for b in range(scores.size(0)):
        la = int(maskA[b].sum().item())
        lb = int(maskB[b].sum().item())
        k = max(1, min(la, lb))
        valid = maskA[b].unsqueeze(1) & maskB[b].unsqueeze(0)
        flat = scores[b].masked_fill(~valid, -1e9).flatten()
        kk = min(k, flat.numel())
        idx = torch.topk(flat, kk).indices
        vals.append(y2d[b].flatten()[idx].float().mean())
    return float(torch.stack(vals).mean()) if vals else 0.0


def contact_density(batch: Dict) -> float:
    if "y2d" not in batch:
        return 0.0
    y2d = batch["y2d"].detach().cpu()
    maskA = batch.get("maskA", torch.ones(y2d.shape[:2])).detach().cpu() > 0.5
    maskB = batch.get("maskB", torch.ones(y2d.size(0), y2d.size(2))).detach().cpu() > 0.5
    vals = []
    for b in range(y2d.size(0)):
        valid = maskA[b].unsqueeze(1) & maskB[b].unsqueeze(0)
        if valid.any():
            vals.append(y2d[b][valid].float().mean())
    return float(torch.stack(vals).mean()) if vals else 0.0


def dataset_group_keys(ds: Dataset, split_mode: str = "group") -> Optional[List[str]]:
    if split_mode == "random":
        return None
    if isinstance(ds, TriagePairDataset):
        pairs = []
        for _, r in ds.rows.iterrows():
            a = str(r[ds.a_col])
            b = str(r[ds.b_col])
            pairs.append((a, b, str(r[ds.pair_col]) if ds.pair_col else "__".join(sorted([a, b]))))
        if split_mode == "protein_component":
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for a, b, _ in pairs:
                union(a, b)
            return [find(a) for a, _, _ in pairs]
        keys = []
        for a, b, pair_id in pairs:
            keys.append(pair_id if split_mode == "group" else "|".join(sorted([a, b])))
        return keys
    if isinstance(ds, TriageSiteDataset):
        keys = []
        for pair_id, _, _, _ in ds.pairs:
            pair_s = str(pair_id)
            keys.append(pair_s.split("_")[0].split("-")[0])
        return keys
    return None


def split_dataset(ds: Dataset, cfg: TRIAGEConfig, seed: int, split_mode: str = "group") -> Tuple[Dataset, Dataset]:
    n = len(ds)
    if n <= 1:
        return ds, ds
    n_val = max(int(round(n * float(cfg.val_fraction))), int(cfg.min_val_items))
    n_val = min(max(1, n_val), n - 1)
    keys = dataset_group_keys(ds, split_mode)
    if keys and len(keys) == n and len(set(keys)) > 1:
        rng = random.Random(seed)
        group_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            group_to_indices[str(key)].append(i)
        groups = list(group_to_indices)
        rng.shuffle(groups)
        val_idx = []
        train_idx = []
        for group in groups:
            target = val_idx if len(val_idx) < n_val else train_idx
            target.extend(group_to_indices[group])
        if len(val_idx) > 0 and len(train_idx) > 0:
            return Subset(ds, train_idx), Subset(ds, val_idx)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)


def split_dataset_train_val_test(ds: Dataset, cfg: TRIAGEConfig, seed: int, split_mode: str = "group") -> Tuple[Dataset, Dataset, Dataset]:
    n = len(ds)
    if n <= 2:
        return ds, ds, ds
    n_val = max(int(round(n * float(cfg.val_fraction))), int(cfg.min_val_items))
    n_test = max(int(round(n * float(getattr(cfg, "test_fraction", 0.10)))), int(cfg.min_val_items))
    n_val = min(max(1, n_val), n - 2)
    n_test = min(max(1, n_test), n - n_val - 1)
    if isinstance(ds, TriagePairDataset) and split_mode == "protein_disjoint":
        uids = []
        for _, r in ds.rows.iterrows():
            uids.append(str(r[ds.a_col]))
            uids.append(str(r[ds.b_col]))
        uid_list = sorted(set(uids))
        rng = random.Random(seed)
        rng.shuffle(uid_list)
        n_uid = len(uid_list)
        n_uid_test = max(1, int(round(n_uid * float(getattr(cfg, "test_fraction", 0.10)))))
        n_uid_val = max(1, int(round(n_uid * float(cfg.val_fraction))))
        test_uids = set(uid_list[:n_uid_test])
        val_uids = set(uid_list[n_uid_test:n_uid_test + n_uid_val])
        train_uids = set(uid_list[n_uid_test + n_uid_val:])
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        dropped = 0
        for i, r in ds.rows.iterrows():
            a = str(r[ds.a_col])
            b = str(r[ds.b_col])
            if a in train_uids and b in train_uids:
                train_idx.append(int(i))
            elif a in val_uids and b in val_uids:
                val_idx.append(int(i))
            elif a in test_uids and b in test_uids:
                test_idx.append(int(i))
            else:
                dropped += 1
        print(
            f"[split-drop] mode=protein_disjoint kept={len(train_idx) + len(val_idx) + len(test_idx)}/{n} "
            f"dropped_cross_split={dropped}",
            flush=True,
        )
        if train_idx and val_idx and test_idx:
            return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)
        print("[split-drop][warn] protein_disjoint produced an empty split; falling back to random split.", flush=True)
    if isinstance(ds, TriagePairDataset) and split_mode == "protein_component":
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        row_pairs: List[Tuple[str, str]] = []
        for _, r in ds.rows.iterrows():
            a = str(r[ds.a_col])
            b = str(r[ds.b_col])
            row_pairs.append((a, b))
            union(a, b)
        component_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, (a, b) in enumerate(row_pairs):
            component_to_indices[find(a)].append(i)
        comps = list(component_to_indices)
        rng = random.Random(seed)
        rng.shuffle(comps)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        for comp in sorted(comps, key=lambda c: len(component_to_indices[c]), reverse=True):
            choices = [
                (len(test_idx) / max(1, n_test), test_idx, n_test),
                (len(val_idx) / max(1, n_val), val_idx, n_val),
                (len(train_idx) / max(1, n - n_val - n_test), train_idx, n - n_val - n_test),
            ]
            choices.sort(key=lambda x: x[0])
            target = choices[0][1]
            target.extend(component_to_indices[comp])
        if train_idx and val_idx and test_idx:
            return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)
        largest = max((len(component_to_indices[comp]) for comp in comps), default=0)
        print(
            f"[split-component][warn] protein_component could not form three non-empty splits "
            f"(components={len(comps)} largest_pairs={largest}/{n}); falling back to a non-disjoint split. "
            "Do not describe the resulting evaluation as protein-disjoint.",
            flush=True,
        )
    keys = dataset_group_keys(ds, split_mode)
    if keys and len(keys) == n and len(set(keys)) > 1:
        rng = random.Random(seed)
        group_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            group_to_indices[str(key)].append(i)
        groups = list(group_to_indices)
        rng.shuffle(groups)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        for group in groups:
            target = test_idx if len(test_idx) < n_test else val_idx if len(val_idx) < n_val else train_idx
            target.extend(group_to_indices[group])
        if train_idx and val_idx and test_idx:
            return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=gen)


def subset_indices(ds: Dataset) -> List[int]:
    if isinstance(ds, Subset):
        return [int(i) for i in ds.indices]
    return list(range(len(ds)))


def pair_subset_uids(base: TriagePairDataset, ds: Dataset) -> set:
    out = set()
    for idx in subset_indices(ds):
        r = base.rows.iloc[int(idx)]
        out.add(str(r[base.a_col]))
        out.add(str(r[base.b_col]))
    return out


def pair_label_counts(base: TriagePairDataset, ds: Dataset) -> Tuple[int, int]:
    pos = 0
    neg = 0
    for idx in subset_indices(ds):
        r = base.rows.iloc[int(idx)]
        pair_id = str(r[base.pair_col]) if base.pair_col else f"{r[base.a_col]}__{r[base.b_col]}"
        if base.y_col:
            y = float(r[base.y_col])
        elif getattr(base, "contact_pair_labels", None):
            y = float(base.contact_pair_labels.get(pair_id, 0.0))
        else:
            y = float(pair_id in base.positive_pair_ids)
        if y >= 0.5:
            pos += 1
        else:
            neg += 1
    return pos, neg


def balance_pair_subset(base: TriagePairDataset, ds: Dataset, seed: int, ratio: float = 1.0) -> Dataset:
    indices = subset_indices(ds)
    pos_idx: List[int] = []
    neg_idx: List[int] = []
    for idx in indices:
        r = base.rows.iloc[int(idx)]
        pair_id = str(r[base.pair_col]) if base.pair_col else f"{r[base.a_col]}__{r[base.b_col]}"
        if base.y_col:
            y = float(r[base.y_col])
        elif getattr(base, "contact_pair_labels", None):
            y = float(base.contact_pair_labels.get(pair_id, 0.0))
        else:
            y = float(pair_id in base.positive_pair_ids)
        (pos_idx if y >= 0.5 else neg_idx).append(int(idx))
    if not pos_idx or not neg_idx:
        return ds
    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    max_pos = min(len(pos_idx), max(1, int(round(len(neg_idx) * ratio))))
    keep = pos_idx[:max_pos] + neg_idx
    rng.shuffle(keep)
    return Subset(base, keep)


def make_loaders(cfg: TRIAGEConfig, args) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    if args.synthetic_debug:
        n = args.debug_samples if args.stage == "debug" else args.samples
        struct = SyntheticTRIAGEDataset(n=n, d_res=cfg.d_res_in, task="struct")
        pair = SyntheticTRIAGEDataset(n=n, d_res=cfg.d_res_in, task="pair")
        print(f"[data] synthetic debug datasets struct={len(struct)} pair={len(pair)}", flush=True)
    elif args.stage in TUNA_PAIR_STAGES and all(
        [args.tuna_train_fourpack_dir, args.tuna_val_fourpack_dir, args.tuna_test_fourpack_dir]
    ):
        def fixed_tuna_dataset(fourpack_dir: str) -> TriagePairDataset:
            level_cfg = copy.deepcopy(cfg)
            level_cfg.pair_fourpack_dir = str(Path(fourpack_dir))
            level_cfg.pair_esm_dir = str(Path(fourpack_dir) / "emb" / "esm2")
            return TriagePairDataset(level_cfg, max_items=args.max_items if args.max_items > 0 else 0)

        pair_train_base = fixed_tuna_dataset(args.tuna_train_fourpack_dir)
        pair_val = fixed_tuna_dataset(args.tuna_val_fourpack_dir)
        pair_test = fixed_tuna_dataset(args.tuna_test_fourpack_dir)
        pair_train: Dataset = pair_train_base
        if args.balance_pair_train:
            before = len(pair_train)
            pair_train = balance_pair_subset(
                pair_train_base,
                pair_train,
                args.seed + 409,
                ratio=args.balance_pair_ratio,
            )
            pos, neg = pair_label_counts(pair_train_base, pair_train)
            print(
                f"[label-balance] fixed_TUnA_train kept={len(pair_train)}/{before} "
                f"pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}",
                flush=True,
            )

        train_uids = pair_subset_uids(pair_train_base, pair_train)
        val_uids = pair_subset_uids(pair_val, pair_val)
        test_uids = pair_subset_uids(pair_test, pair_test)
        overlap = {
            "train_val": len(train_uids & val_uids),
            "train_test": len(train_uids & test_uids),
            "val_test": len(val_uids & test_uids),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Fixed TUnA levels have protein overlap: {overlap}")
        print(
            f"[split-fixed][TUnA] Intra1/train={len(pair_train)} "
            f"Intra0/validation={len(pair_val)} Intra2/test={len(pair_test)} "
            f"protein_overlap={overlap}",
            flush=True,
        )
        train_kw = dict(
            batch_size=args.batch_size or cfg.batch_size,
            shuffle=True,
            collate_fn=collate_pad,
            num_workers=cfg.num_workers,
        )
        val_kw = dict(
            batch_size=args.batch_size or cfg.batch_size,
            shuffle=False,
            collate_fn=collate_pad,
            num_workers=cfg.num_workers,
        )
        train_loader = DataLoader(pair_train, **train_kw)
        val_loader = DataLoader(pair_val, **val_kw)
        test_loader = DataLoader(pair_test, **val_kw)
        return train_loader, train_loader, val_loader, val_loader, test_loader, test_loader
    elif args.stage in TUNA_PAIR_STAGES and any(
        [args.tuna_train_fourpack_dir, args.tuna_val_fourpack_dir, args.tuna_test_fourpack_dir]
    ):
        raise ValueError(
            "Fixed TUnA evaluation requires all three options: "
            "--tuna-train-fourpack-dir, --tuna-val-fourpack-dir, and "
            "--tuna-test-fourpack-dir."
        )
    elif args.stage in RBP400_TRIAGE_STAGES:
        rbp_root = args.rbp400_root or cfg.rbp400_root
        id_list_path = args.rbp400_id_list or getattr(cfg, "rbp400_id_list", "")
        l1_train = GraphRBFPPDataset(cfg, rbp_root, "train", max_items=args.max_items if args.max_items > 0 else 0, esm_dir=cfg.rbp400_esm_dir, id_list_path=id_list_path, dataset_label="RBP400")
        l1_val = GraphRBFPPDataset(cfg, rbp_root, "val", max_items=0, esm_dir=cfg.rbp400_esm_dir, id_list_path=id_list_path, dataset_label="RBP400")
        l1_test = GraphRBFPPDataset(cfg, rbp_root, "test", max_items=0, esm_dir=cfg.rbp400_esm_dir, id_list_path=id_list_path, dataset_label="RBP400")
        pair = TriagePairDataset(cfg, max_items=args.max_items if args.max_items > 0 else 0)
        pair_train, pair_val, pair_test = split_dataset_train_val_test(pair, cfg, args.seed + 29, args.split_mode)
        if isinstance(pair, TriagePairDataset):
            for name, ds_part in (("pair_train_raw", pair_train), ("pair_val", pair_val), ("pair_test", pair_test)):
                pos, neg = pair_label_counts(pair, ds_part)
                print(f"[label-balance] {name} pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}", flush=True)
            if args.balance_pair_train:
                before = len(pair_train)
                pair_train = balance_pair_subset(pair, pair_train, args.seed + 409, ratio=args.balance_pair_ratio)
                pos, neg = pair_label_counts(pair, pair_train)
                print(
                    f"[label-balance] pair_train_balanced kept={len(pair_train)}/{before} "
                    f"pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}",
                    flush=True,
                )
        print(
            f"[data] RBP400 triage train_l1={len(l1_train)} val_l1={len(l1_val)} test_l1={len(l1_test)} "
            f"pair_train={len(pair_train)} pair_val={len(pair_val)} pair_test={len(pair_test)} rbp_root={rbp_root}",
            flush=True,
        )
        print_l1_label_balance("RBP400", Path(rbp_root), "train", l1_train.pids, float(getattr(cfg, "l1_pos_weight_cap", 12.0)))
        print_l1_label_balance("RBP400", Path(rbp_root), "val", l1_val.pids, float(getattr(cfg, "l1_pos_weight_cap", 12.0)))
        l1_batch_size = int(getattr(cfg, "l1_batch_size", cfg.batch_size))
        l1_workers = int(getattr(cfg, "l1_num_workers", cfg.num_workers))
        pair_batch_size = args.batch_size or cfg.batch_size
        train_l1_loader = DataLoader(l1_train, batch_size=l1_batch_size, shuffle=True, collate_fn=collate_pad, num_workers=l1_workers)
        val_l1_loader = DataLoader(l1_val, batch_size=l1_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=l1_workers)
        test_l1_loader = DataLoader(l1_test, batch_size=l1_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=l1_workers)
        pair_train_loader = DataLoader(pair_train, batch_size=pair_batch_size, shuffle=True, collate_fn=collate_pad, num_workers=cfg.num_workers)
        pair_val_loader = DataLoader(pair_val, batch_size=pair_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=cfg.num_workers)
        pair_test_loader = DataLoader(pair_test, batch_size=pair_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=cfg.num_workers)
        return train_l1_loader, pair_train_loader, val_l1_loader, pair_val_loader, test_l1_loader, pair_test_loader
    elif args.stage in DEST_TRIAGE_STAGES:
        dest_root = args.pp_root or getattr(cfg, "dest_root", cfg.pp_root)
        esm_dir = getattr(cfg, "dest_esm_dir", cfg.pp_esm_dir)
        train_ds = DestTriagePairDataset(cfg, dest_root, "train", max_items=args.max_items if args.max_items > 0 else 0, esm_dir=esm_dir)
        val_ds = DestTriagePairDataset(cfg, dest_root, "val", max_items=0, esm_dir=esm_dir)
        test_ds = DestTriagePairDataset(cfg, dest_root, "test", max_items=0, esm_dir=esm_dir)
        batch_size = int(getattr(cfg, "l1_batch_size", cfg.batch_size))
        workers = int(getattr(cfg, "l1_num_workers", cfg.num_workers))
        print(
            f"[data] Dest tri-level fusion train_pairs={len(train_ds)} val_pairs={len(val_ds)} "
            f"test_pairs={len(test_ds)} root={dest_root}",
            flush=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad, num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=workers)
        return train_loader, train_loader, val_loader, val_loader, test_loader, test_loader
    elif args.stage in L1_STAGES:
        pp_root = args.pp_root or cfg.pp_root
        esm_dir = cfg.pp_esm_dir
        id_list_path = None
        dataset_label = "GraphRBF-PP"
        if args.stage == "l1_dest":
            pp_root = args.pp_root or getattr(cfg, "dest_root", cfg.pp_root)
            esm_dir = getattr(cfg, "dest_esm_dir", cfg.pp_esm_dir)
            dataset_label = "Dest"
        if args.stage in RBP400_STAGES:
            pp_root = args.rbp400_root or cfg.rbp400_root
            esm_dir = cfg.rbp400_esm_dir
            id_list_path = args.rbp400_id_list or getattr(cfg, "rbp400_id_list", "")
            dataset_label = "RBP400"
        train_ds = GraphRBFPPDataset(cfg, pp_root, "train", max_items=args.max_items if args.max_items > 0 else 0, esm_dir=esm_dir, id_list_path=id_list_path, dataset_label=dataset_label)
        val_ds = GraphRBFPPDataset(cfg, pp_root, "val", max_items=0, esm_dir=esm_dir, id_list_path=id_list_path, dataset_label=dataset_label)
        test_ds = GraphRBFPPDataset(cfg, pp_root, "test", max_items=0, esm_dir=esm_dir, id_list_path=id_list_path, dataset_label=dataset_label)
        print(
            f"[data] {dataset_label} L1 train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} root={pp_root}",
            flush=True,
        )
        if args.stage in RBP400_STAGES:
            balance_cap = float(getattr(cfg, "l1_pos_weight_cap", 12.0))
            print_l1_label_balance(dataset_label, Path(pp_root), "train", train_ds.pids, balance_cap)
            print_l1_label_balance(dataset_label, Path(pp_root), "val", val_ds.pids, balance_cap)
            print_l1_label_balance(dataset_label, Path(pp_root), "test", test_ds.pids, balance_cap)
            print_l1_structure_coverage(dataset_label, Path(pp_root), "train", train_ds.pids)
            print_l1_structure_coverage(dataset_label, Path(pp_root), "val", val_ds.pids)
            print_l1_structure_coverage(dataset_label, Path(pp_root), "test", test_ds.pids)
            loss_parts = [
                f"per_protein={bool(getattr(cfg, 'l1_per_protein_loss', False))}",
                f"label_smoothing={float(getattr(cfg, 'l1_label_smoothing', 0.0)):.3f}",
                f"extreme_label_weight={float(getattr(cfg, 'l1_extreme_label_weight', 1.0)):.3f}",
                f"pos_weight_cap={balance_cap:.2f}",
                f"rank_w={float(getattr(cfg, 'w_l1_rank', 0.0)):.3f}",
                f"rank_margin={float(getattr(cfg, 'l1_rank_margin', 0.0)):.3f}",
                f"rank_start={int(getattr(cfg, 'l1_rank_start_epoch', 1))}",
                f"rank_ramp={int(getattr(cfg, 'l1_rank_ramp_epochs', 1))}",
                f"rank_max_pairs={int(getattr(cfg, 'l1_rank_max_pairs', 2048))}",
                f"hard_rank_w={float(getattr(cfg, 'w_l1_hard_rank', 0.0)):.3f}",
                f"hard_rank_start={int(getattr(cfg, 'l1_hard_rank_start_epoch', 1))}",
                f"topband_w={float(getattr(cfg, 'w_l1_topband_bce', 0.0)):.3f}",
                f"topband_start={int(getattr(cfg, 'l1_topband_start_epoch', 1))}",
                f"l10_boundary_w={float(getattr(cfg, 'w_l1_l10_boundary', 0.0)):.3f}",
                f"l10_boundary_start={int(getattr(cfg, 'l1_l10_boundary_start_epoch', 1))}",
                f"l10_boundary_margin={float(getattr(cfg, 'l1_l10_boundary_margin', 0.0)):.3f}",
                f"binary_score_w=(acc:{float(getattr(cfg, 'l1_score_w_acc', 0.0)):.2f},"
                f"auc:{float(getattr(cfg, 'l1_score_w_auc', 0.0)):.2f},"
                f"auprc:{float(getattr(cfg, 'l1_score_w_auprc', 0.0)):.2f},"
                f"mcc:{float(getattr(cfg, 'l1_score_w_mcc', 0.0)):.2f},"
                f"f1:{float(getattr(cfg, 'l1_score_w_f1', 0.0)):.2f},"
                f"loss_penalty:{float(getattr(cfg, 'l1_score_loss_penalty', 0.0)):.2f})",
                f"thr_mode={str(getattr(cfg, 'l1_threshold_mode', 'auto_mcc'))}",
                f"thr_min_recall={float(getattr(cfg, 'l1_threshold_min_recall', 0.0)):.2f}",
                f"AGER={bool(getattr(cfg, 'l1_ager_enable', False))}",
                f"ager_alpha={float(getattr(cfg, 'l1_ager_alpha', 0.0)):.2f}",
                f"ager_radius={float(getattr(cfg, 'l1_ager_radius', 0.0)):.1f}",
                f"ager_top_m={int(getattr(cfg, 'l1_ager_top_m', 0))}",
                f"raw_skip={bool(getattr(cfg, 'use_l1_raw_skip', False))}",
                f"multiscale={bool(getattr(cfg, 'use_l1_multiscale_head', False))}",
                f"geom_adapter={bool(getattr(cfg, 'use_l1_geom_adapter', False))}",
                "ager_mode=boost_only",
            ]
            if float(getattr(cfg, "l1_zero_label_weight", -1.0)) >= 0:
                loss_parts.append(f"zero_label_weight={float(getattr(cfg, 'l1_zero_label_weight')):.3f}")
            if float(getattr(cfg, "l1_full_label_weight", -1.0)) >= 0:
                loss_parts.append(f"full_label_weight={float(getattr(cfg, 'l1_full_label_weight')):.3f}")
            if bool(getattr(cfg, "l1_exclude_zero_label_proteins", False)):
                loss_parts.append("exclude_zero_label=True")
            print(f"[loss][RBP400] {' '.join(loss_parts)}", flush=True)
        l1_batch_size = int(getattr(cfg, "l1_batch_size", cfg.batch_size))
        l1_workers = int(getattr(cfg, "l1_num_workers", cfg.num_workers))
        print(f"[loader] l1 batch_size={l1_batch_size} num_workers={l1_workers}", flush=True)
        train_loader = DataLoader(train_ds, batch_size=l1_batch_size, shuffle=True, collate_fn=collate_pad, num_workers=l1_workers)
        val_loader = DataLoader(val_ds, batch_size=l1_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=l1_workers)
        test_loader = DataLoader(test_ds, batch_size=l1_batch_size, shuffle=False, collate_fn=collate_pad, num_workers=l1_workers)
        return train_loader, train_loader, val_loader, val_loader, test_loader, test_loader
    else:
        max_items = args.max_items if args.max_items > 0 else 0
        struct = TriageSiteDataset(cfg, max_items=max_items)
        pair = TriagePairDataset(cfg, max_items=max_items)
        print(f"[data] real datasets struct={len(struct)} pair={len(pair)}", flush=True)
        print(
            f"[features] pair: seq+PSSM+DSSP_SS, pair_DSSP_RSA={'on' if cfg.use_pair_dssp_rsa else 'off'}; "
            f"site: seq+PSSM+DSSP_SS+RSA; external annotations=off",
            flush=True,
        )
        print(f"[length] max_pair_len={cfg.max_pair_len} max_site_len={cfg.max_site_len}", flush=True)
    struct_train, struct_val, struct_test = split_dataset_train_val_test(struct, cfg, args.seed + 11, args.split_mode)
    pair_train, pair_val, pair_test = split_dataset_train_val_test(pair, cfg, args.seed + 29, args.split_mode)
    if args.split_mode == "random":
        print(
            "[split][warn] random split can leak proteins/pairs across train/val; "
            "use --split-mode group or --split-mode protein_component for formal validation.",
            flush=True,
        )
    print(
        f"[split] mode={args.split_mode} struct_train={len(struct_train)} struct_val={len(struct_val)} "
        f"struct_test={len(struct_test)} pair_train={len(pair_train)} pair_val={len(pair_val)} "
        f"pair_test={len(pair_test)}",
        flush=True,
    )
    if isinstance(pair, TriagePairDataset):
        for name, ds_part in (("pair_train_raw", pair_train), ("pair_val", pair_val), ("pair_test", pair_test)):
            pos, neg = pair_label_counts(pair, ds_part)
            print(f"[label-balance] {name} pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}", flush=True)
        if args.balance_pair_train:
            before = len(pair_train)
            pair_train = balance_pair_subset(pair, pair_train, args.seed + 409, ratio=args.balance_pair_ratio)
            pos, neg = pair_label_counts(pair, pair_train)
            print(
                f"[label-balance] pair_train_balanced kept={len(pair_train)}/{before} "
                f"pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}",
                flush=True,
            )
        train_uids = pair_subset_uids(pair, pair_train)
        val_uids = pair_subset_uids(pair, pair_val)
        test_uids = pair_subset_uids(pair, pair_test)
        print(
            f"[split-leak-check] pair train_val_uid_overlap={len(train_uids & val_uids)} "
            f"train_test_uid_overlap={len(train_uids & test_uids)} "
            f"val_test_uid_overlap={len(val_uids & test_uids)}",
            flush=True,
        )
    train_kw = dict(batch_size=args.batch_size or cfg.batch_size, shuffle=True, collate_fn=collate_pad, num_workers=cfg.num_workers)
    val_kw = dict(batch_size=args.batch_size or cfg.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=cfg.num_workers)
    return (
        DataLoader(struct_train, **train_kw),
        DataLoader(pair_train, **train_kw),
        DataLoader(struct_val, **val_kw),
        DataLoader(pair_val, **val_kw),
        DataLoader(struct_test, **val_kw),
        DataLoader(pair_test, **val_kw),
    )


def stage_epochs(cfg: TRIAGEConfig, stage: str) -> int:
    return {
        "debug": cfg.epochs_debug,
        "struct_pretrain": cfg.epochs_struct_pretrain,
        "pair_fusion": cfg.epochs_pair_fusion,
        "tuna_pair_finetune": cfg.epochs_pair_fusion,
        "joint_finetune": cfg.epochs_joint_finetune,
        "l1_dest": cfg.epochs_l1_graphrbf,
        "dest_triage": cfg.epochs_l1_graphrbf,
        "l1_graphrbf": cfg.epochs_l1_graphrbf,
        "l1_rbp400": cfg.epochs_l1_graphrbf,
        "l1_rbp400_topk": cfg.epochs_l1_graphrbf,
        "rbp400_triage": cfg.epochs_l1_graphrbf,
    }[stage]


def stage_patience(cfg: TRIAGEConfig, stage: str) -> int:
    return {
        "debug": cfg.patience_debug,
        "struct_pretrain": cfg.patience_struct_pretrain,
        "pair_fusion": cfg.patience_pair_fusion,
        "tuna_pair_finetune": cfg.patience_pair_fusion,
        "joint_finetune": cfg.patience_joint_finetune,
        "l1_dest": cfg.patience_l1_graphrbf,
        "dest_triage": cfg.patience_l1_graphrbf,
        "l1_graphrbf": cfg.patience_l1_graphrbf,
        "l1_rbp400": cfg.patience_l1_graphrbf,
        "l1_rbp400_topk": cfg.patience_l1_graphrbf,
        "rbp400_triage": cfg.patience_l1_graphrbf,
    }[stage]


def apply_epoch_override(cfg: TRIAGEConfig, stage: str, epochs: int):
    if epochs <= 0:
        return
    if stage == "debug":
        cfg.epochs_debug = int(epochs)
    elif stage == "struct_pretrain":
        cfg.epochs_struct_pretrain = int(epochs)
    elif stage in ("pair_fusion", "tuna_pair_finetune"):
        cfg.epochs_pair_fusion = int(epochs)
    elif stage == "joint_finetune":
        cfg.epochs_joint_finetune = int(epochs)
    elif stage in L1_STAGES or stage in RBP400_TRIAGE_STAGES or stage in DEST_TRIAGE_STAGES:
        cfg.epochs_l1_graphrbf = int(epochs)


def save_checkpoint(path: str, model: TRIAGEPPIModel, cfg: TRIAGEConfig, stage: str, epoch: int, best_metric: float):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "stage": stage,
        "epoch": epoch,
        "best_metric": best_metric,
        "feature_spec": feature_spec(cfg),
        "threshold": 0.5,
        "topk": cfg.topk,
        "topm": cfg.topm,
    }
    torch.save(ckpt, path)


def clone_state_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def init_ema_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return clone_state_cpu(model)


def update_ema_state(model: torch.nn.Module, ema_state: Dict[str, torch.Tensor], decay: float) -> None:
    state = model.state_dict()
    for key, value in state.items():
        v = value.detach().cpu()
        if key not in ema_state:
            ema_state[key] = v.clone()
        elif torch.is_floating_point(v):
            ema_state[key].mul_(float(decay)).add_(v.to(dtype=ema_state[key].dtype), alpha=1.0 - float(decay))
        else:
            ema_state[key] = v.clone()


def load_state_cpu(model: torch.nn.Module, state: Dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)


def add_state_to_average(avg: Optional[Dict[str, torch.Tensor]], state: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
    if avg is None or n <= 0:
        return {k: v.clone() for k, v in state.items()}
    out = {}
    for key, value in avg.items():
        if torch.is_floating_point(value):
            out[key] = value + (state[key].to(dtype=value.dtype) - value) / float(n + 1)
        else:
            out[key] = state[key].clone()
    return out


def rbp400_pareto_snapshot_candidate(stats: Dict, epoch: int) -> bool:
    return (
        int(epoch) >= 10
        and float(stats.get("l1_recall_l5", 0.0)) >= 0.25
        and float(stats.get("l1_recall_l10", 0.0)) >= 0.135
        and float(stats.get("l1_precision_10", 0.0)) >= 0.53
        and float(stats.get("l1_hit_20", 0.0)) >= 0.88
    )


def read_checkpoint_score(path: str, device: torch.device) -> Tuple[float, int]:
    if not path or not os.path.exists(path):
        return -float("inf"), 0
    try:
        ckpt = torch.load(path, map_location=device)
        return float(ckpt.get("best_metric", -float("inf"))), int(ckpt.get("epoch", 0))
    except Exception as exc:
        print(f"[best] cannot read existing checkpoint score: {path} | {exc}", flush=True)
        return -float("inf"), 0


def load_resume(model: TRIAGEPPIModel, resume: str, device: torch.device):
    if not resume:
        return
    ckpt = torch.load(resume, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    current = model.state_dict()
    compatible = {}
    skipped = []

    def legacy_crossblock_key(key: str) -> List[str]:
        if ".na." in key:
            return [key.replace(".na.", ".nqA."), key.replace(".na.", ".nkA.")]
        if ".nb." in key:
            return [key.replace(".nb.", ".nqB."), key.replace(".nb.", ".nkB.")]
        return [key]

    for key, value in state.items():
        mapped = legacy_crossblock_key(key)
        loaded_any = False
        for out_key in mapped:
            if out_key in current and current[out_key].shape == value.shape:
                compatible[out_key] = value
                loaded_any = True
        if not loaded_any:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"[resume] loaded={resume} missing={len(missing)} unexpected={len(unexpected)} "
        f"skipped_shape={len(skipped)}",
        flush=True,
    )
    if skipped:
        print(f"[resume-skipped-sample] {skipped[:12]}", flush=True)
    if missing:
        print(f"[resume-missing-sample] {list(missing)[:12]}", flush=True)
    return list(missing), list(unexpected), skipped


def set_requires_grad_by_stage(model: TRIAGEPPIModel, stage: str, epoch: int, freeze_pair_local_epochs: int) -> str:
    for p in model.parameters():
        p.requires_grad = True
    if stage in L1_STAGES:
        train_prefixes = ["projA", "filmA", "encA", "res_head_A", "res_vec", "res_logit"]
        if bool(getattr(model.cfg, "use_l1_geom_early", False)):
            train_prefixes.append("l1_geom_embed_A")
        if bool(getattr(model.cfg, "use_l1_raw_skip", False)):
            train_prefixes.append("l1_raw_head_A")
        if bool(getattr(model.cfg, "use_l1_geom_adapter", False)):
            train_prefixes.append("l1_geom_adapter_A")
        train_prefixes = tuple(train_prefixes)
        trainable = 0
        frozen = 0
        for name, p in model.named_parameters():
            keep = name.startswith(train_prefixes)
            p.requires_grad = keep
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        return f"l1_finetune trainable_params={trainable} frozen_params={frozen}"
    if stage not in ("pair_fusion", "tuna_pair_finetune") or freeze_pair_local_epochs <= 0 or epoch > freeze_pair_local_epochs:
        return "all_trainable"
    frozen_prefixes = (
        "projA", "projB", "filmA", "filmB", "encA", "encB", "cross",
        "res_head_A", "res_head_B", "res_vec", "res_logit",
        "l2_projA", "l2_projB", "l2_refine", "interface_vec", "interface_logit",
    )
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            p.requires_grad = False
            n_frozen += p.numel()
    return f"pair_head_warmup_frozen_params={n_frozen}"


def ckpt_name_for_stage(stage: str, topk_engineering: bool = False) -> str:
    if topk_engineering and stage in TOPK_ENGINEERING_CKPT:
        return TOPK_ENGINEERING_CKPT[stage]
    return STAGE_CKPT[stage]


def default_resume_for_stage(stage: str, out_dir: str, topk_engineering: bool = False) -> str:
    if stage == "l1_dest":
        return getattr(dest_data_config(), "dest_base_checkpoint")
    if stage in DEST_TRIAGE_STAGES:
        return getattr(dest_triage_config(), "dest_base_checkpoint")
    if stage == "l1_rbp400_topk":
        return os.path.join(out_dir, TOPK_ENGINEERING_CKPT["joint_finetune"])
    if stage in RBP400_TRIAGE_STAGES:
        return os.path.join(out_dir, TOPK_ENGINEERING_CKPT["joint_finetune"])
    if stage in TUNA_PAIR_STAGES:
        return getattr(tuna_pair_finetune_config(), "tuna_base_checkpoint")
    if stage in L1_STAGES:
        return os.path.join(out_dir, STAGE_CKPT["joint_finetune"])
    if stage == "pair_fusion":
        return os.path.join(out_dir, ckpt_name_for_stage("struct_pretrain", topk_engineering))
    if stage == "joint_finetune":
        return os.path.join(out_dir, ckpt_name_for_stage("pair_fusion", topk_engineering))
    return ""


def iter_stage_batches(stage: str, struct_loader: DataLoader, pair_loader: DataLoader, cfg: TRIAGEConfig):
    if stage in ("debug", "struct_pretrain"):
        for b in struct_loader:
            yield "structure", b
    elif stage in DEST_TRIAGE_STAGES:
        for b in struct_loader:
            yield "structure", b
    elif stage in L1_STAGES:
        for b in struct_loader:
            yield "l1", b
    elif stage in RBP400_TRIAGE_STAGES:
        l1_iter = itertools.cycle(struct_loader)
        pair_iter = itertools.cycle(pair_loader)
        steps = max(len(struct_loader), len(pair_loader))
        for _ in range(steps):
            for _ in range(cfg.joint_struct_steps):
                yield "l1", next(l1_iter)
            for _ in range(cfg.joint_pair_steps):
                yield "pair", next(pair_iter)
    elif stage in ("pair_fusion", "tuna_pair_finetune"):
        for b in pair_loader:
            yield "pair", b
    elif stage == "joint_finetune":
        s_iter = itertools.cycle(struct_loader)
        p_iter = itertools.cycle(pair_loader)
        steps = max(len(struct_loader), len(pair_loader))
        for _ in range(steps):
            for _ in range(cfg.joint_struct_steps):
                yield "structure", next(s_iter)
            for _ in range(cfg.joint_pair_steps):
                yield "pair", next(p_iter)
    else:
        raise ValueError(stage)


def collect_epoch_stats(
    model: TRIAGEPPIModel,
    batches,
    cfg: TRIAGEConfig,
    device: torch.device,
    epoch: int,
    train_mode: bool,
    opt=None,
    max_steps: int = 0,
    progress_label: str = "",
    progress_every: int = 100,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    ema_decay: float = 0.0,
) -> Dict:
    losses = []
    log_sums = defaultdict(float)
    log_counts = defaultdict(int)
    probs, labels = [], []
    l1_probs, l1_labels = [], []
    l1_topk_sums = defaultdict(float)
    l1_topk_n = 0.0
    l1_raw_topk_sums = defaultdict(float)
    l1_raw_topk_n = 0.0
    l2_precisions = []
    l2_topl_precisions = []
    contact_densities = []
    gate_sums = torch.zeros(3)
    rel_vals = []
    n_gate = 0
    nan_count = 0

    model.train(train_mode)
    grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_ctx:
        for step, (task, batch) in enumerate(batches, start=1):
            if max_steps and max_steps > 0 and step > max_steps:
                break
            batch = batch_to_device(batch, device)
            if train_mode:
                opt.zero_grad(set_to_none=True)
            out = forward_batch(model, batch)
            loss, log = compute_joint_loss(out, batch, task, cfg, epoch)
            if not torch.isfinite(loss):
                print(f"[skip-nan] ep={epoch} step={step} task={task} log={log}", flush=True)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                nan_count += 1
                if nan_count > 20:
                    raise FloatingPointError(f"too many non-finite losses at epoch {epoch}: {nan_count}")
                continue
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                if ema_state is not None and float(ema_decay) > 0:
                    update_ema_state(model, ema_state, float(ema_decay))
            losses.append(float(loss.detach().cpu()))
            for key, value in log.items():
                if key == "loss":
                    continue
                try:
                    log_sums[key] += float(value)
                    log_counts[key] += 1
                except Exception:
                    pass
            if task == "pair" or "y2d" in batch or "y_pair" in batch:
                y = batch["y_pair"] if task == "pair" else ((batch["y2d"].flatten(1).sum(1) > 0).float())
                probs.append(out["p_triage"].detach().cpu())
                labels.append(y.detach().cpu())
            l2_precisions.append(l2_precision_at_topm(out, batch))
            l2_topl_precisions.append(l2_topl_precision(out, batch))
            contact_densities.append(contact_density(batch))
            if "y_res_A" in batch:
                m = batch.get("maskA", torch.ones_like(batch["y_res_A"])) > 0.5
                raw_topk = residue_topk_metrics(out["p_res_A"], batch["y_res_A"], m)
                n_raw_topk = float(raw_topk.get("l1_topk_n", 0.0))
                if n_raw_topk > 0:
                    for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20"):
                        l1_raw_topk_sums[key] += float(raw_topk[key]) * n_raw_topk
                    l1_raw_topk_n += n_raw_topk
                p_res_A_metric = apply_l1_ager(out["p_res_A"], batch.get("coordsA"), m, cfg)
                l1_probs.append(p_res_A_metric.detach().cpu()[m.detach().cpu()])
                l1_labels.append(batch["y_res_A"].detach().cpu()[m.detach().cpu()])
                topk = residue_topk_metrics(p_res_A_metric, batch["y_res_A"], m)
                n_topk = float(topk.get("l1_topk_n", 0.0))
                if n_topk > 0:
                    for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20"):
                        l1_topk_sums[key] += float(topk[key]) * n_topk
                    l1_topk_n += n_topk
            if "y_res_B" in batch:
                m = batch.get("maskB", torch.ones_like(batch["y_res_B"])) > 0.5
                p_res_B_metric = out["p_res_B"]
                l1_probs.append(p_res_B_metric.detach().cpu()[m.detach().cpu()])
                l1_labels.append(batch["y_res_B"].detach().cpu()[m.detach().cpu()])
                topk = residue_topk_metrics(p_res_B_metric, batch["y_res_B"], m)
                n_topk = float(topk.get("l1_topk_n", 0.0))
                if n_topk > 0:
                    for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20"):
                        l1_topk_sums[key] += float(topk[key]) * n_topk
                    l1_topk_n += n_topk
            gates = out["fusion_weights"]
            gate_sums += torch.tensor([
                float(gates["g_res"].mean().detach().cpu()),
                float(gates["g_interface"].mean().detach().cpu()),
                float(gates["g_pair"].mean().detach().cpu()),
            ])
            rel_vals.append(float(out["evidence_reliability"].mean().detach().cpu()))
            n_gate += 1
            if progress_label and progress_every and progress_every > 0 and step % progress_every == 0:
                print(
                    f"[progress] {progress_label} ep={epoch} step={step} "
                    f"loss={sum(losses) / max(1, len(losses)):.4f}",
                    flush=True,
                )

    if probs:
        all_prob = torch.cat(probs)
        all_lab = torch.cat(labels)
        bm = binary_metrics(all_prob, all_lab)
        if all_lab.numel() == 0 or torch.unique(all_lab).numel() < 2:
            bm["auprc"] = float("nan")
            bm["auroc"] = float("nan")
            bm["mcc"] = float("nan")
    else:
        bm = {"auprc": float("nan"), "auroc": float("nan"), "mcc": float("nan"), "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if l1_probs:
        l1_bm = binary_metrics_best_threshold(
            torch.cat(l1_probs),
            torch.cat(l1_labels),
            beta=float(getattr(cfg, "l1_threshold_beta", 1.0)),
            min_precision=float(getattr(cfg, "l1_threshold_min_precision", 0.0)),
            min_recall=float(getattr(cfg, "l1_threshold_min_recall", 0.0)),
            mode=str(getattr(cfg, "l1_threshold_mode", "auto_mcc")),
        )
    else:
        l1_bm = {"auprc": 0.0, "auroc": float("nan"), "mcc": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "thr": 0.5}
    l1_auprc = l1_bm["auprc"]
    means = gate_sums / max(1, n_gate)
    mean_l2_topm = sum(l2_precisions) / max(1, len(l2_precisions))
    mean_l2_topl = sum(l2_topl_precisions) / max(1, len(l2_topl_precisions))
    mean_density = sum(contact_densities) / max(1, len(contact_densities))
    mean_l1_topk = {
        key: l1_topk_sums[key] / max(1.0, l1_topk_n)
        for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20")
    }
    mean_l1_raw_topk = {
        f"raw_{key}": l1_raw_topk_sums[key] / max(1.0, l1_raw_topk_n)
        for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20")
    }
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "pair_auprc": bm["auprc"],
        "pair_auroc": bm["auroc"],
        "pair_mcc": bm["mcc"],
        "acc": bm["acc"],
        "precision": bm["precision"],
        "recall": bm["recall"],
        "f1": bm["f1"],
        "l1_auprc": l1_auprc,
        "l1_auroc": l1_bm["auroc"],
        "l1_mcc": l1_bm["mcc"],
        "l1_acc": l1_bm["acc"],
        "l1_precision": l1_bm["precision"],
        "l1_recall": l1_bm["recall"],
        "l1_f1": l1_bm["f1"],
        "l1_thr": l1_bm.get("thr", 0.5),
        **mean_l1_topk,
        **mean_l1_raw_topk,
        "l1_topk_n": l1_topk_n,
        "l2_topm_precision": mean_l2_topm,
        "l2_topl_precision": mean_l2_topl,
        "contact_density": mean_density,
        "l2_topm_enrichment": mean_l2_topm / max(mean_density, 1e-8),
        "l2_topl_enrichment": mean_l2_topl / max(mean_density, 1e-8),
        "mean_g_res": float(means[0]),
        "mean_g_interface": float(means[1]),
        "mean_g_pair": float(means[2]),
        "mean_evidence_reliability": sum(rel_vals) / max(1, len(rel_vals)),
        **{
            f"mean_{key}": log_sums[key] / max(1, log_counts[key])
            for key in log_sums
        },
    }


def collect_l1_probability_ensemble_stats(
    models: List[TRIAGEPPIModel],
    batches,
    cfg: TRIAGEConfig,
    device: torch.device,
) -> Dict:
    l1_probs, l1_labels = [], []
    l1_topk_sums = defaultdict(float)
    l1_topk_n = 0.0
    l1_raw_topk_sums = defaultdict(float)
    l1_raw_topk_n = 0.0
    for model in models:
        model.eval()
    with torch.no_grad():
        for _, batch in batches:
            batch = batch_to_device(batch, device)
            probs_a = []
            for model in models:
                out = forward_batch(model, batch)
                probs_a.append(out["p_res_A"])
            p_raw = torch.stack(probs_a, dim=0).mean(dim=0)
            if "y_res_A" not in batch:
                continue
            m = batch.get("maskA", torch.ones_like(batch["y_res_A"])) > 0.5
            raw_topk = residue_topk_metrics(p_raw, batch["y_res_A"], m)
            n_raw_topk = float(raw_topk.get("l1_topk_n", 0.0))
            if n_raw_topk > 0:
                for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20"):
                    l1_raw_topk_sums[key] += float(raw_topk[key]) * n_raw_topk
                l1_raw_topk_n += n_raw_topk
            p_metric = apply_l1_ager(p_raw, batch.get("coordsA"), m, cfg)
            l1_probs.append(p_metric.detach().cpu()[m.detach().cpu()])
            l1_labels.append(batch["y_res_A"].detach().cpu()[m.detach().cpu()])
            topk = residue_topk_metrics(p_metric, batch["y_res_A"], m)
            n_topk = float(topk.get("l1_topk_n", 0.0))
            if n_topk > 0:
                for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20"):
                    l1_topk_sums[key] += float(topk[key]) * n_topk
                l1_topk_n += n_topk
    if l1_probs:
        l1_bm = binary_metrics_best_threshold(
            torch.cat(l1_probs),
            torch.cat(l1_labels),
            beta=float(getattr(cfg, "l1_threshold_beta", 1.0)),
            min_precision=float(getattr(cfg, "l1_threshold_min_precision", 0.0)),
            min_recall=float(getattr(cfg, "l1_threshold_min_recall", 0.0)),
            mode=str(getattr(cfg, "l1_threshold_mode", "auto_mcc")),
        )
    else:
        l1_bm = {"auprc": 0.0, "auroc": float("nan"), "mcc": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "thr": 0.5}
    mean_l1_topk = {
        key: l1_topk_sums[key] / max(1.0, l1_topk_n)
        for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20")
    }
    mean_l1_raw_topk = {
        f"raw_{key}": l1_raw_topk_sums[key] / max(1.0, l1_raw_topk_n)
        for key in ("l1_recall_l5", "l1_recall_l10", "l1_precision_10", "l1_hit_2", "l1_hit_20")
    }
    return {
        "loss": 0.0,
        "l1_auprc": l1_bm["auprc"],
        "l1_auroc": l1_bm["auroc"],
        "l1_mcc": l1_bm["mcc"],
        "l1_acc": l1_bm["acc"],
        "l1_precision": l1_bm["precision"],
        "l1_recall": l1_bm["recall"],
        "l1_f1": l1_bm["f1"],
        "l1_thr": l1_bm.get("thr", 0.5),
        **mean_l1_topk,
        **mean_l1_raw_topk,
        "l1_topk_n": l1_topk_n,
    }


def validation_score(stage: str, stats: Dict, cfg: Optional[TRIAGEConfig] = None, loss_penalty: float = 0.0) -> float:
    if stage in DEST_TRIAGE_STAGES:
        w_acc = float(getattr(cfg, "l1_score_w_acc", 0.10)) if cfg is not None else 0.10
        w_auc = float(getattr(cfg, "l1_score_w_auc", 0.30)) if cfg is not None else 0.30
        w_auprc = float(getattr(cfg, "l1_score_w_auprc", 0.25)) if cfg is not None else 0.25
        w_mcc = float(getattr(cfg, "l1_score_w_mcc", 0.25)) if cfg is not None else 0.25
        w_f1 = float(getattr(cfg, "l1_score_w_f1", 0.10)) if cfg is not None else 0.10
        return float(
            w_acc * stats["l1_acc"]
            + w_auc * stats["l1_auroc"]
            + w_auprc * stats["l1_auprc"]
            + w_mcc * max(0.0, stats["l1_mcc"])
            + w_f1 * stats["l1_f1"]
            - loss_penalty * stats["loss"]
        )
    if stage in RBP400_TRIAGE_STAGES:
        gates = [float(stats.get("mean_g_res", 0.0)), float(stats.get("mean_g_interface", 0.0)), float(stats.get("mean_g_pair", 0.0))]
        gate_entropy = 0.0
        for g in gates:
            if g > 1e-8:
                gate_entropy -= g * math.log(g)
        gate_entropy = gate_entropy / math.log(3.0)
        effective_loss_penalty = float(loss_penalty)
        if effective_loss_penalty <= 0.0:
            effective_loss_penalty = float(getattr(cfg, "rbp400_triage_score_loss_penalty", 0.01)) if cfg is not None else 0.01
        pair_auprc = 0.0 if math.isnan(float(stats.get("pair_auprc", float("nan")))) else float(stats.get("pair_auprc", 0.0))
        pair_mcc = 0.0 if math.isnan(float(stats.get("pair_mcc", float("nan")))) else max(0.0, float(stats.get("pair_mcc", 0.0)))
        w_pair_auprc = float(getattr(cfg, "rbp400_triage_score_w_pair_auprc", 0.05)) if cfg is not None else 0.05
        w_pair_mcc = float(getattr(cfg, "rbp400_triage_score_w_pair_mcc", 0.03)) if cfg is not None else 0.03
        w_recall_l5 = float(getattr(cfg, "rbp400_triage_score_w_recall_l5", 0.30)) if cfg is not None else 0.30
        w_recall_l10 = float(getattr(cfg, "rbp400_triage_score_w_recall_l10", 0.30)) if cfg is not None else 0.30
        w_precision_10 = float(getattr(cfg, "rbp400_triage_score_w_precision_10", 0.25)) if cfg is not None else 0.25
        w_hit_20 = float(getattr(cfg, "rbp400_triage_score_w_hit_20", 0.05)) if cfg is not None else 0.05
        w_gate_entropy = float(getattr(cfg, "rbp400_triage_score_w_gate_entropy", 0.02)) if cfg is not None else 0.02
        return float(
            w_pair_auprc * pair_auprc
            + w_pair_mcc * pair_mcc
            + w_recall_l5 * float(stats.get("l1_recall_l5", 0.0))
            + w_recall_l10 * float(stats.get("l1_recall_l10", 0.0))
            + w_precision_10 * float(stats.get("l1_precision_10", 0.0))
            + w_hit_20 * float(stats.get("l1_hit_20", 0.0))
            + w_gate_entropy * gate_entropy
            - effective_loss_penalty * float(stats.get("loss", 0.0))
        )
    if stage in TOPK_STAGES:
        effective_loss_penalty = float(loss_penalty)
        if effective_loss_penalty <= 0.0:
            effective_loss_penalty = float(getattr(cfg, "l1_score_loss_penalty", 0.0))
        return float(
            float(getattr(cfg, "l1_score_w_recall_l5", 0.45)) * stats["l1_recall_l5"]
            + float(getattr(cfg, "l1_score_w_recall_l10", 0.35)) * stats["l1_recall_l10"]
            + float(getattr(cfg, "l1_score_w_precision_10", 0.10)) * stats["l1_precision_10"]
            + float(getattr(cfg, "l1_score_w_hit_20", 0.10)) * stats["l1_hit_20"]
            + float(getattr(cfg, "l1_score_w_hit_2", 0.0)) * stats["l1_hit_2"]
            - effective_loss_penalty * stats["loss"]
        )
    if stage == "l1_rbp400":
        w_auc = float(getattr(cfg, "l1_score_w_auc", 0.35))
        w_auprc = float(getattr(cfg, "l1_score_w_auprc", 0.45))
        w_mcc = float(getattr(cfg, "l1_score_w_mcc", 0.12))
        w_f1 = float(getattr(cfg, "l1_score_w_f1", 0.06))
        w_acc = float(getattr(cfg, "l1_score_w_acc", 0.02))
        effective_loss_penalty = float(loss_penalty)
        if effective_loss_penalty <= 0.0:
            effective_loss_penalty = float(getattr(cfg, "l1_score_loss_penalty", 0.0))
        return float(
            w_acc * stats["l1_acc"]
            + w_auc * stats["l1_auroc"]
            + w_auprc * stats["l1_auprc"]
            + w_mcc * max(0.0, stats["l1_mcc"])
            + w_f1 * stats["l1_f1"]
            - effective_loss_penalty * stats["loss"]
        )
    if stage == "l1_dest":
        w_auc = float(getattr(cfg, "l1_score_w_auc", 0.25))
        w_auprc = float(getattr(cfg, "l1_score_w_auprc", 0.25))
        w_mcc = float(getattr(cfg, "l1_score_w_mcc", 0.15))
        w_f1 = float(getattr(cfg, "l1_score_w_f1", 0.05))
        w_recall_l5 = float(getattr(cfg, "l1_score_w_recall_l5", 0.15))
        w_recall_l10 = float(getattr(cfg, "l1_score_w_recall_l10", 0.10))
        w_precision_10 = float(getattr(cfg, "l1_score_w_precision_10", 0.05))
        return float(
            w_auc * stats["l1_auroc"]
            + w_auprc * stats["l1_auprc"]
            + w_mcc * max(0.0, stats["l1_mcc"])
            + w_f1 * stats["l1_f1"]
            + w_recall_l5 * stats.get("l1_recall_l5", 0.0)
            + w_recall_l10 * stats.get("l1_recall_l10", 0.0)
            + w_precision_10 * stats.get("l1_precision_10", 0.0)
            - loss_penalty * stats["loss"]
        )
    if stage == "l1_graphrbf":
        w_auc = float(getattr(cfg, "l1_score_w_auc", 0.70))
        w_auprc = float(getattr(cfg, "l1_score_w_auprc", 0.10))
        w_mcc = float(getattr(cfg, "l1_score_w_mcc", 0.15))
        w_f1 = float(getattr(cfg, "l1_score_w_f1", 0.05))
        return float(
            w_auc * stats["l1_auroc"]
            + w_auprc * stats["l1_auprc"]
            + w_mcc * max(0.0, stats["l1_mcc"])
            + w_f1 * stats["l1_f1"]
            - loss_penalty * stats["loss"]
        )
    if stage in ("debug", "struct_pretrain"):
        gate_penalty = max(0.0, float(stats["mean_g_pair"]) - 0.50)
        return float(stats["l1_auprc"] + 2.0 * stats["l2_topm_precision"] + 2.0 * stats["l2_topl_precision"] - 0.05 * stats["loss"] - 0.5 * gate_penalty)
    if stage in ("pair_fusion", "tuna_pair_finetune"):
        return float(stats["pair_auprc"] + max(0.0, stats["pair_mcc"]) - loss_penalty * stats["loss"])
    return float(
        stats["pair_auprc"]
        + max(0.0, stats["pair_mcc"])
        + stats["l2_topm_precision"]
        + stats["l1_auprc"]
        - loss_penalty * stats["loss"]
    )


def rbp400_target_balance_score(stats: Dict) -> float:
    """Score closeness to the old ARISE-PPI top-k row without hiding weak metrics."""
    targets = {
        "l1_recall_l5": 0.2979,
        "l1_recall_l10": 0.1870,
        "l1_precision_10": 0.6479,
        "l1_hit_20": 0.8571,
    }
    ratios = []
    for key, target in targets.items():
        ratios.append(float(stats.get(key, 0.0)) / max(float(target), 1e-8))
    min_ratio = min(ratios) if ratios else 0.0
    mean_capped = sum(min(1.0, r) for r in ratios) / max(1, len(ratios))
    return float(0.70 * min_ratio + 0.30 * mean_capped)


def rbp400_sidecar_values(stats: Dict, score: float) -> Dict[str, float]:
    return {
        "score": float(score),
        "acc": float(stats.get("l1_acc", 0.0)),
        "precision": float(stats.get("l1_precision", 0.0)),
        "recall": float(stats.get("l1_recall", 0.0)),
        "f1": float(stats.get("l1_f1", 0.0)),
        "auroc": float(stats.get("l1_auroc", 0.0)),
        "auprc": float(stats.get("l1_auprc", 0.0)),
        "mcc": float(stats.get("l1_mcc", 0.0)),
    }


def rbp400_topk_sidecar_values(stats: Dict, score: float) -> Dict[str, float]:
    return {
        "score": float(score),
        "target": rbp400_target_balance_score(stats),
        "recall_l5": float(stats.get("l1_recall_l5", 0.0)),
        "recall_l10": float(stats.get("l1_recall_l10", 0.0)),
        "precision_10": float(stats.get("l1_precision_10", 0.0)),
        "p10_r10": 0.5 * float(stats.get("l1_precision_10", 0.0)) + 0.5 * float(stats.get("l1_recall_l10", 0.0)),
    }


def format_rbp400_metrics(stats: Dict, raw_digits: int = 4, include_binary: bool = True) -> str:
    text = (
        f"ACC={stats['l1_acc']:.4f} "
        f"Precision={stats['l1_precision']:.4f} "
        f"Recall={stats['l1_recall']:.4f} "
        f"F1={stats['l1_f1']:.4f} "
        f"AUROC={stats['l1_auroc']:.4f} "
        f"AUPRC={stats['l1_auprc']:.4f} "
        f"MCC={stats['l1_mcc']:.4f} "
        f"thr={stats['l1_thr']:.3f}"
    )
    if include_binary:
        text += (
            " topk("
        f"R@L/5={stats['l1_recall_l5']:.4f} "
        f"R@L/10={stats['l1_recall_l10']:.4f} "
        f"P@10={stats['l1_precision_10']:.4f} "
        f"Hit@20={stats['l1_hit_20']:.4f} "
        f"Hit@2={stats['l1_hit_2']:.4f} "
        f"raw=({stats.get('raw_l1_recall_l5', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_recall_l10', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_precision_10', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_hit_20', float('nan')):.{raw_digits}f}) "
            f"proteins={int(stats['l1_topk_n'])})"
        )
    return text


def format_rbp400_topk_metrics(stats: Dict, raw_digits: int = 4) -> str:
    return (
        f"topk(R@L/5={stats['l1_recall_l5']:.4f} "
        f"R@L/10={stats['l1_recall_l10']:.4f} "
        f"P@10={stats['l1_precision_10']:.4f} "
        f"Hit@20={stats['l1_hit_20']:.4f} "
        f"Hit@2={stats['l1_hit_2']:.4f} "
        f"raw=({stats.get('raw_l1_recall_l5', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_recall_l10', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_precision_10', float('nan')):.{raw_digits}f},"
        f"{stats.get('raw_l1_hit_20', float('nan')):.{raw_digits}f}) "
        f"proteins={int(stats['l1_topk_n'])})"
    )


def format_rbp400_triage_metrics(stats: Dict, raw_digits: int = 4) -> str:
    if float(stats.get("contact_density", 0.0)) <= 0.0:
        l2_text = "l2(TopM=NA TopL=NA enrichM=NA; no_y2d_contact_labels)"
    else:
        l2_text = (
            f"l2(TopM={stats['l2_topm_precision']:.4f} TopL={stats['l2_topl_precision']:.4f} "
            f"enrichM={stats['l2_topm_enrichment']:.2f})"
        )
    return (
        f"{format_rbp400_topk_metrics(stats, raw_digits=raw_digits)} "
        f"pair(AUPRC={stats['pair_auprc']:.4f} AUROC={stats['pair_auroc']:.4f} MCC={stats['pair_mcc']:.4f}) "
        f"{l2_text} "
        f"gate=({stats['mean_g_res']:.2f},{stats['mean_g_interface']:.2f},{stats['mean_g_pair']:.2f})"
    )


def format_rbp400_stage_metrics(stage: str, stats: Dict, raw_digits: int = 4, full_triage: bool = True) -> str:
    if stage in RBP400_TRIAGE_STAGES:
        return format_rbp400_triage_metrics(stats, raw_digits=raw_digits) if full_triage else format_rbp400_topk_metrics(stats, raw_digits=raw_digits)
    if stage in TOPK_STAGES:
        return format_rbp400_topk_metrics(stats, raw_digits=raw_digits)
    if stage == "l1_rbp400":
        return format_rbp400_metrics(stats, raw_digits=raw_digits, include_binary=False)
    return format_rbp400_metrics(stats, raw_digits=raw_digits, include_binary=True)


def resolve_step_limits(stage: str, args) -> Tuple[int, int]:
    if args.full_epoch:
        return 0, 0
    train_steps = args.train_max_steps
    val_steps = args.val_max_steps
    if train_steps < 0:
        train_steps = 1500 if stage in ("pair_fusion", "tuna_pair_finetune") else 1200 if stage == "joint_finetune" or stage in RBP400_TRIAGE_STAGES or stage in DEST_TRIAGE_STAGES else 0
    if val_steps < 0:
        val_steps = 400 if stage in ("pair_fusion", "tuna_pair_finetune") else 300 if stage == "joint_finetune" or stage in RBP400_TRIAGE_STAGES or stage in DEST_TRIAGE_STAGES else 0
    return int(train_steps), int(val_steps)


def append_metrics(path: str, row: Dict):
    exists = os.path.exists(path)
    fieldnames = list(row.keys())
    if exists:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            old_fieldnames = list(reader.fieldnames or [])
            if old_fieldnames and any(k not in old_fieldnames for k in fieldnames):
                rows = []
                for old_row in reader:
                    # Historical metrics.csv files may contain rows with more
                    # cells than the old header; DictReader stores those under
                    # key None. Drop them while preserving named columns.
                    rows.append({k: old_row.get(k, "") for k in old_fieldnames})
                merged = old_fieldnames + [k for k in fieldnames if k not in old_fieldnames]
                rows.append(row)
                with open(path, "w", newline="", encoding="utf-8") as wf:
                    w = csv.DictWriter(wf, fieldnames=merged, delimiter=",")
                    w.writeheader()
                    w.writerows(rows)
                return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        if not exists:
            w.writeheader()
        w.writerow(row)


def make_scheduler(opt, stage: str, total_epochs: int, cfg: Optional[TRIAGEConfig] = None):
    if (stage in L1_STAGES or stage in RBP400_TRIAGE_STAGES or stage in DEST_TRIAGE_STAGES) and cfg is not None:
        warmup_epochs = int(getattr(cfg, "l1_warmup_epochs", 2))
        total_epochs = min(int(total_epochs), int(getattr(cfg, "l1_scheduler_epochs", total_epochs)))
        min_factor = float(getattr(cfg, "l1_min_lr_factor", 0.20))
    else:
        warmup_epochs = 8 if stage == "struct_pretrain" else 3
        min_factor = 0.0
    we = min(int(warmup_epochs), max(1, int(total_epochs) // 8))

    def lr_lambda(ep):
        if ep < we:
            return float(ep + 1) / float(max(we, 1))
        t = min(1.0, float(ep - we) / float(max(int(total_epochs) - we, 1)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(min_factor + (1.0 - min_factor) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train_one_stage(args):
    topk_engineering = bool(getattr(args, "topk_engineering", False))
    if topk_engineering:
        cfg = topk_engineering_data_config()
    elif args.stage == "l1_rbp400":
        cfg = rbp400_data_config()
    elif args.stage == "l1_rbp400_topk":
        cfg = rbp400_topk_data_config()
    elif args.stage in RBP400_TRIAGE_STAGES:
        cfg = rbp400_triage_data_config()
    elif args.stage in DEST_TRIAGE_STAGES:
        cfg = dest_triage_config()
    elif args.stage == "l1_dest":
        cfg = dest_data_config()
    elif args.stage in TUNA_PAIR_STAGES:
        cfg = tuna_pair_finetune_config()
        if args.split_mode == "protein_disjoint":
            args.split_mode = str(getattr(cfg, "tuna_split_mode", "protein_component"))
        if args.seed == 42:
            args.seed = int(getattr(cfg, "tuna_seed", 42))
    else:
        cfg = current_data_config()
    if args.d_res_in > 0:
        cfg.d_res_in = args.d_res_in
    if args.d_model > 0:
        cfg.d_model = args.d_model
    if args.topk > 0:
        cfg.topk = args.topk
    if args.topm > 0:
        cfg.topm = args.topm
    if args.batch_size > 0:
        cfg.batch_size = args.batch_size
        if args.stage in L1_STAGES or args.stage in RBP400_TRIAGE_STAGES:
            cfg.l1_batch_size = args.batch_size
    if args.l1_batch_size > 0:
        cfg.l1_batch_size = args.l1_batch_size
    if args.l1_lr > 0:
        cfg.l1_lr = args.l1_lr
    if args.l1_pos_weight_cap > 0:
        cfg.l1_pos_weight_cap = args.l1_pos_weight_cap
    if args.w_l1_rank >= 0:
        cfg.w_l1_rank = args.w_l1_rank
    if args.l1_rank_margin > 0:
        cfg.l1_rank_margin = args.l1_rank_margin
    if args.w_l1_hard_rank >= 0:
        cfg.w_l1_hard_rank = args.w_l1_hard_rank
    if args.l1_hard_rank_start > 0:
        cfg.l1_hard_rank_start_epoch = args.l1_hard_rank_start
    if args.l1_hard_rank_ramp > 0:
        cfg.l1_hard_rank_ramp_epochs = args.l1_hard_rank_ramp
    if args.w_l1_topband_bce >= 0:
        cfg.w_l1_topband_bce = args.w_l1_topband_bce
    if args.l1_topband_start > 0:
        cfg.l1_topband_start_epoch = args.l1_topband_start
    if args.l1_topband_ramp > 0:
        cfg.l1_topband_ramp_epochs = args.l1_topband_ramp
    if args.l1_topband_frac > 0:
        cfg.l1_topband_frac = args.l1_topband_frac
    if args.w_l1_l10_boundary >= 0:
        cfg.w_l1_l10_boundary = args.w_l1_l10_boundary
    if args.l1_l10_boundary_start > 0:
        cfg.l1_l10_boundary_start_epoch = args.l1_l10_boundary_start
    if args.l1_l10_boundary_ramp > 0:
        cfg.l1_l10_boundary_ramp_epochs = args.l1_l10_boundary_ramp
    if args.l1_l10_boundary_margin > 0:
        cfg.l1_l10_boundary_margin = args.l1_l10_boundary_margin
    if args.l1_extreme_label_weight > 0:
        cfg.l1_extreme_label_weight = args.l1_extreme_label_weight
    if args.l1_zero_label_weight >= 0:
        cfg.l1_zero_label_weight = args.l1_zero_label_weight
    if args.l1_full_label_weight >= 0:
        cfg.l1_full_label_weight = args.l1_full_label_weight
    if args.include_zero_label_proteins:
        cfg.l1_exclude_zero_label_proteins = False
    if args.l1_ager_alpha >= 0:
        cfg.l1_ager_alpha = args.l1_ager_alpha
    if args.l1_ager_radius > 0:
        cfg.l1_ager_radius = args.l1_ager_radius
    if args.l1_ager_top_m > 0:
        cfg.l1_ager_top_m = args.l1_ager_top_m
    if args.num_workers >= 0:
        cfg.num_workers = args.num_workers
    if args.l1_num_workers >= 0:
        cfg.l1_num_workers = args.l1_num_workers
    if args.val_fraction > 0:
        cfg.val_fraction = args.val_fraction
    if args.test_fraction > 0:
        cfg.test_fraction = args.test_fraction
    if args.max_pair_len > 0:
        cfg.max_pair_len = args.max_pair_len
    if args.max_site_len > 0:
        cfg.max_site_len = args.max_site_len
    if args.pp_root:
        cfg.pp_root = args.pp_root
    if args.rbp400_root:
        cfg.rbp400_root = args.rbp400_root
    if args.rbp400_id_list:
        cfg.rbp400_id_list = args.rbp400_id_list
    if args.stage in RBP400_TRIAGE_STAGES:
        print(f"[esm-cache] RBP400 triage uses RBP400 ESM dir={cfg.rbp400_esm_dir} and pair ESM dir={cfg.pair_esm_dir}", flush=True)
    elif args.stage not in L1_STAGES and args.stage not in DEST_TRIAGE_STAGES:
        ensure_esm_cache(cfg, args)
    else:
        if args.stage not in DEST_TRIAGE_STAGES:
            cfg.w_struct_gate_regularization = 0.0
        if args.stage == "l1_dest" or args.stage in DEST_TRIAGE_STAGES:
            ensure_dest_prepared(cfg)
            pp_root_backup, pp_esm_backup = cfg.pp_root, cfg.pp_esm_dir
            cfg.pp_root = getattr(cfg, "dest_root", cfg.pp_root)
            cfg.pp_esm_dir = getattr(cfg, "dest_esm_dir", cfg.pp_esm_dir)
            ensure_pp_esm_cache(cfg, args)
            cfg.pp_root, cfg.pp_esm_dir = pp_root_backup, pp_esm_backup
        elif args.stage == "l1_graphrbf":
            ensure_pp_esm_cache(cfg, args)
        else:
            print(f"[esm-cache] RBP400 uses prepared features/optional cached ESM dir={cfg.rbp400_esm_dir}", flush=True)
    apply_epoch_override(cfg, args.stage, args.epochs)
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TRIAGEPPIModel(cfg).to(device)

    out_dir = args.out_dir or cfg.run_dir
    if (args.stage in TUNA_PAIR_STAGES or args.stage == "l1_dest" or args.stage in DEST_TRIAGE_STAGES) and args.out_dir == DEFAULT_OUT_DIR:
        out_dir = cfg.run_dir
    os.makedirs(out_dir, exist_ok=True)
    resume = args.resume or default_resume_for_stage(args.stage, out_dir, topk_engineering=topk_engineering)
    if args.stage == "l1_rbp400_topk" and not args.resume and resume and not os.path.exists(resume):
        raise FileNotFoundError(
            f"Top-k engineering base not found: {resume}. "
            "Run `python model/train.py --stage topk_engineering --reset-best` first."
        )
    if resume and os.path.exists(resume):
        load_resume(model, resume, device)
    elif resume:
        print(f"[resume] expected checkpoint not found, starting fresh: {resume}", flush=True)

    opt_lr = float(getattr(cfg, "l1_lr", cfg.lr)) if args.stage in L1_STAGES or args.stage in RBP400_TRIAGE_STAGES or args.stage in DEST_TRIAGE_STAGES else float(cfg.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=opt_lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(opt, args.stage, stage_epochs(cfg, args.stage), cfg) if args.use_scheduler else None
    struct_loader, pair_loader, struct_val_loader, pair_val_loader, struct_test_loader, pair_test_loader = make_loaders(cfg, args)

    ckpt_path = os.path.join(out_dir, ckpt_name_for_stage(args.stage, topk_engineering=topk_engineering))
    if (args.stage in L1_STAGES or args.stage in RBP400_TRIAGE_STAGES or args.stage in DEST_TRIAGE_STAGES) and args.max_items and args.max_items > 0:
        ckpt_path = os.path.join(out_dir, f"triage_{args.stage}_smoke.pt")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    best_metric, best_epoch = read_checkpoint_score(ckpt_path, device)
    if args.reset_best:
        if math.isfinite(best_metric):
            print(
                f"[best] reset requested; preserving existing checkpoint for comparison "
                f"path={ckpt_path} metric={best_metric:.6f} epoch={best_epoch}",
                flush=True,
            )
        else:
            print(f"[best] reset requested; no existing checkpoint, starting fresh path={ckpt_path}", flush=True)
            best_metric = -1.0
            best_epoch = 0
    elif math.isfinite(best_metric):
        print(f"[best] existing checkpoint={ckpt_path} metric={best_metric:.6f} epoch={best_epoch}", flush=True)
    else:
        best_metric = -1.0
        best_epoch = 0
    train_max_steps, val_max_steps = resolve_step_limits(args.stage, args)
    patience = args.patience if args.patience is not None and args.patience >= 0 else stage_patience(cfg, args.stage)
    if train_max_steps or val_max_steps:
        print(
            f"[epoch-limit] train_max_steps={train_max_steps or 'all'} "
            f"val_max_steps={val_max_steps or 'all'}; use --full-epoch for full data pass",
            flush=True,
        )

    stale_epochs = 0
    run_best_metric = -float("inf")
    run_best_epoch = 0
    last_trainability = ""
    pareto_avg_state: Optional[Dict[str, torch.Tensor]] = None
    pareto_avg_n = 0
    pareto_epochs: List[int] = []
    ema_decay = float(getattr(cfg, "l1_ema_decay", 0.0)) if args.stage in RBP400_STAGES else 0.0
    ema_state: Optional[Dict[str, torch.Tensor]] = init_ema_state(model) if ema_decay > 0 else None
    if ema_state is not None:
        print(f"[ema] enabled decay={ema_decay:.4f} validation/checkpoints use EMA weights", flush=True)
    rbp400_sidecar_paths: Dict[str, str] = {}
    rbp400_sidecar_best: Dict[str, float] = {}
    rbp400_sidecar_epochs: Dict[str, int] = {}
    if args.stage in RBP400_STAGES:
        if args.stage == "l1_rbp400_topk":
            sidecar_names = ("score", "target", "recall_l5", "recall_l10", "precision_10", "p10_r10")
            sidecar_prefix = "triage_l1_rbp400_topk_best"
        elif args.stage in RBP400_TRIAGE_STAGES:
            sidecar_names = ("score", "target", "recall_l5", "recall_l10", "precision_10", "p10_r10")
            sidecar_prefix = "triage_rbp400_triage_best"
        else:
            sidecar_names = ("score", "acc", "precision", "recall", "f1", "auroc", "auprc", "mcc")
            sidecar_prefix = "triage_l1_rbp400_binary_best"
        for name in sidecar_names:
            sidecar_path = os.path.join(out_dir, f"{sidecar_prefix}_{name}.pt")
            rbp400_sidecar_paths[name] = sidecar_path
            sidecar_metric, sidecar_epoch = read_checkpoint_score(sidecar_path, device)
            if math.isfinite(sidecar_metric):
                rbp400_sidecar_best[name] = sidecar_metric
                rbp400_sidecar_epochs[name] = sidecar_epoch
                print(
                    f"[sidecar] existing name={name} metric={sidecar_metric:.6f} "
                    f"epoch={sidecar_epoch} path={sidecar_path}",
                    flush=True,
                )
            else:
                rbp400_sidecar_best[name] = -float("inf")
                rbp400_sidecar_epochs[name] = 0
    for ep in range(1, stage_epochs(cfg, args.stage) + 1):
        trainability = set_requires_grad_by_stage(model, args.stage, ep, args.freeze_pair_local_epochs)
        if trainability != last_trainability:
            print(f"[trainability] ep={ep} {trainability}", flush=True)
            last_trainability = trainability
        train_stats = collect_epoch_stats(
            model,
            iter_stage_batches(args.stage, struct_loader, pair_loader, cfg),
            cfg,
            device,
            ep,
            train_mode=True,
            opt=opt,
            max_steps=train_max_steps,
            progress_label=f"{args.stage}/train",
            progress_every=args.progress_every,
            ema_state=ema_state,
            ema_decay=ema_decay,
        )
        train_weight_state = None
        if ema_state is not None:
            train_weight_state = clone_state_cpu(model)
            load_state_cpu(model, ema_state, device)
        val_stats = collect_epoch_stats(
            model,
            iter_stage_batches(args.stage, struct_val_loader, pair_val_loader, cfg),
            cfg,
            device,
            ep,
            train_mode=False,
            max_steps=val_max_steps,
            progress_label=f"{args.stage}/val",
            progress_every=args.progress_every,
        )
        score = validation_score(args.stage, val_stats, cfg, args.score_loss_penalty)
        sidecar_saved: List[str] = []
        if rbp400_sidecar_best:
            sidecar_values = rbp400_topk_sidecar_values(val_stats, score) if args.stage == "l1_rbp400_topk" or args.stage in RBP400_TRIAGE_STAGES else rbp400_sidecar_values(val_stats, score)
            for name, value in sidecar_values.items():
                if value > rbp400_sidecar_best[name] + args.min_delta:
                    rbp400_sidecar_best[name] = value
                    rbp400_sidecar_epochs[name] = ep
                    save_checkpoint(rbp400_sidecar_paths[name], model, cfg, args.stage, ep, value)
                    sidecar_saved.append(name)
        if args.stage in RBP400_STAGES and rbp400_pareto_snapshot_candidate(val_stats, ep):
            pareto_avg_state = add_state_to_average(pareto_avg_state, clone_state_cpu(model), pareto_avg_n)
            pareto_avg_n += 1
            pareto_epochs.append(ep)
            print(f"[pareto-swa] added ep={ep} n={pareto_avg_n}", flush=True)
        is_run_best = score > run_best_metric + args.min_delta
        if is_run_best:
            run_best_metric = score
            run_best_epoch = ep
            stale_epochs = 0
        else:
            stale_epochs += 1
        is_best = score > best_metric + args.min_delta
        if is_best:
            best_metric = score
            best_epoch = ep
            save_checkpoint(ckpt_path, model, cfg, args.stage, ep, best_metric)
        if train_weight_state is not None:
            load_state_cpu(model, train_weight_state, device)

        row = {
            "stage": args.stage,
            "epoch": ep,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "pair_auprc": val_stats["pair_auprc"],
            "pair_auroc": val_stats["pair_auroc"],
            "pair_mcc": val_stats["pair_mcc"],
            "l1_auprc": val_stats["l1_auprc"],
            "l1_auroc": val_stats["l1_auroc"],
            "l1_mcc": val_stats["l1_mcc"],
            "l1_acc": val_stats["l1_acc"],
            "l1_precision": val_stats["l1_precision"],
            "l1_recall": val_stats["l1_recall"],
            "l1_f1": val_stats["l1_f1"],
            "l1_thr": val_stats["l1_thr"],
            "l1_recall_l5": val_stats["l1_recall_l5"],
            "l1_recall_l10": val_stats["l1_recall_l10"],
            "l1_precision_10": val_stats["l1_precision_10"],
            "l1_hit_2": val_stats["l1_hit_2"],
            "l1_hit_20": val_stats["l1_hit_20"],
            "raw_l1_recall_l5": val_stats.get("raw_l1_recall_l5", float("nan")),
            "raw_l1_recall_l10": val_stats.get("raw_l1_recall_l10", float("nan")),
            "raw_l1_precision_10": val_stats.get("raw_l1_precision_10", float("nan")),
            "raw_l1_hit_20": val_stats.get("raw_l1_hit_20", float("nan")),
            "l1_topk_n": val_stats["l1_topk_n"],
            "l2_topk_precision": val_stats["l2_topm_precision"],
            "l2_topm_precision": val_stats["l2_topm_precision"],
            "l2_topl_precision": val_stats["l2_topl_precision"],
            "contact_density": val_stats["contact_density"],
            "l2_topm_enrichment": val_stats["l2_topm_enrichment"],
            "l2_topl_enrichment": val_stats["l2_topl_enrichment"],
            "mean_g_res": val_stats["mean_g_res"],
            "mean_g_interface": val_stats["mean_g_interface"],
            "mean_g_pair": val_stats["mean_g_pair"],
            "mean_evidence_reliability": val_stats["mean_evidence_reliability"],
            "mean_L_res_A": val_stats.get("mean_L_res_A", float("nan")),
            "mean_L_l1_rank_A": val_stats.get("mean_L_l1_rank_A", float("nan")),
            "mean_L_l1_hard_rank_A": val_stats.get("mean_L_l1_hard_rank_A", float("nan")),
            "mean_w_l1_hard_rank": val_stats.get("mean_w_l1_hard_rank", 0.0),
            "mean_L_l1_l10_boundary_A": val_stats.get("mean_L_l1_l10_boundary_A", float("nan")),
            "mean_w_l1_l10_boundary": val_stats.get("mean_w_l1_l10_boundary", 0.0),
            "rbp400_target_balance": rbp400_target_balance_score(val_stats) if args.stage in RBP400_STAGES else float("nan"),
            "val_score": score,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "is_best": int(is_best),
            "run_best_metric": run_best_metric,
            "run_best_epoch": run_best_epoch,
            "is_run_best": int(is_run_best),
            "stale_epochs": stale_epochs,
            "lr": opt.param_groups[0]["lr"],
        }
        append_metrics(metrics_path, row)
        if args.stage in RBP400_TRIAGE_STAGES:
            metric_text = (
                f"{format_rbp400_triage_metrics(row, raw_digits=3)} "
                f"Lrank={row['mean_L_l1_rank_A']:.3f} "
                f"Lhard={row['mean_L_l1_hard_rank_A']:.3f}@{row['mean_w_l1_hard_rank']:.3f} "
                f"L10bd={row['mean_L_l1_l10_boundary_A']:.3f}@{row['mean_w_l1_l10_boundary']:.3f}"
            )
            if sidecar_saved:
                metric_text += f" sidecar={','.join(sidecar_saved)}"
        elif args.stage in TOPK_STAGES:
            metric_text = (
                f"{format_rbp400_topk_metrics(row, raw_digits=3)} "
                f"Lrank={row['mean_L_l1_rank_A']:.3f} "
                f"Lhard={row['mean_L_l1_hard_rank_A']:.3f}@{row['mean_w_l1_hard_rank']:.3f} "
                f"L10bd={row['mean_L_l1_l10_boundary_A']:.3f}@{row['mean_w_l1_l10_boundary']:.3f}"
            )
            if sidecar_saved:
                metric_text += f" sidecar={','.join(sidecar_saved)}"
        elif args.stage == "l1_rbp400":
            metric_text = (
                f"{format_rbp400_metrics(row, raw_digits=3, include_binary=False)} "
                f"Lrank={row['mean_L_l1_rank_A']:.3f} "
                f"Lhard={row['mean_L_l1_hard_rank_A']:.3f}@{row['mean_w_l1_hard_rank']:.3f} "
                f"L10bd={row['mean_L_l1_l10_boundary_A']:.3f}@{row['mean_w_l1_l10_boundary']:.3f}"
            )
            if sidecar_saved:
                metric_text += f" sidecar={','.join(sidecar_saved)}"
        elif args.stage in DEST_TRIAGE_STAGES:
            metric_text = (
                f"residue_auprc={row['l1_auprc']:.4f} residue_auroc={row['l1_auroc']:.4f} "
                f"residue_mcc={row['l1_mcc']:.4f} "
                f"acc={row['l1_acc']:.4f} precision={row['l1_precision']:.4f} "
                f"recall={row['l1_recall']:.4f} f1={row['l1_f1']:.4f} thr={row['l1_thr']:.3f} "
                f"R@L/5={row['l1_recall_l5']:.4f} R@L/10={row['l1_recall_l10']:.4f} "
                f"P@10={row['l1_precision_10']:.4f} Hit@20={row['l1_hit_20']:.4f} Hit@2={row['l1_hit_2']:.4f}"
            )
        elif args.stage in L1_STAGES:
            metric_text = (
                f"l1_auprc={row['l1_auprc']:.4f} l1_auroc={row['l1_auroc']:.4f} "
                f"l1_mcc={row['l1_mcc']:.4f} "
                f"acc={row['l1_acc']:.4f} precision={row['l1_precision']:.4f} "
                f"recall={row['l1_recall']:.4f} f1={row['l1_f1']:.4f} thr={row['l1_thr']:.3f} "
                f"R@L/5={row['l1_recall_l5']:.4f} R@L/10={row['l1_recall_l10']:.4f} "
                f"P@10={row['l1_precision_10']:.4f} Hit@20={row['l1_hit_20']:.4f} Hit@2={row['l1_hit_2']:.4f}"
            )
        elif args.stage in ("debug", "struct_pretrain"):
            metric_text = (
                f"pair=N/A l1={row['l1_auprc']:.4f} "
                f"l2TopM={row['l2_topm_precision']:.4f} l2TopL={row['l2_topl_precision']:.4f} "
                f"density={row['contact_density']:.4f} "
                f"enrichM={row['l2_topm_enrichment']:.2f} enrichL={row['l2_topl_enrichment']:.2f}"
            )
        else:
            metric_text = f"auprc={row['pair_auprc']:.4f} auroc={row['pair_auroc']:.4f} mcc={row['pair_mcc']:.4f}"
        metric_tag = f"{args.stage}/val" if args.stage in L1_STAGES or args.stage in RBP400_TRIAGE_STAGES or args.stage in DEST_TRIAGE_STAGES else args.stage
        print(
            f"[{metric_tag}] ep={ep} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"{metric_text} score={score:.4f} best={best_metric:.4f}@ep{best_epoch} "
            f"saved={int(is_best)} lr={row['lr']:.2e} "
            f"g=({row['mean_g_res']:.2f},{row['mean_g_interface']:.2f},{row['mean_g_pair']:.2f})",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
        if patience > 0 and stale_epochs >= patience:
            print(
                f"[early-stop] no run-local improvement for {stale_epochs} epochs "
                f"(run_best={run_best_metric:.4f}@ep{run_best_epoch}, "
                f"global_best={best_metric:.4f}@ep{best_epoch})",
                flush=True,
            )
            break

    if args.stage in RBP400_STAGES and pareto_avg_state is not None and pareto_avg_n >= 2:
        current_state = clone_state_cpu(model)
        model.load_state_dict(pareto_avg_state, strict=True)
        swa_stats = collect_epoch_stats(
            model,
            iter_stage_batches(args.stage, struct_val_loader, pair_val_loader, cfg),
            cfg,
            device,
            -1,
            train_mode=False,
            max_steps=val_max_steps,
            progress_label=f"{args.stage}/pareto_swa_val",
            progress_every=args.progress_every,
        )
        swa_score = validation_score(args.stage, swa_stats, cfg, args.score_loss_penalty)
        print(
            f"[pareto-swa] epochs={pareto_epochs} val_score={swa_score:.4f} "
            f"{format_rbp400_stage_metrics(args.stage, swa_stats, raw_digits=4)}",
            flush=True,
        )
        if args.stage == "l1_rbp400_topk":
            swa_name = "triage_l1_rbp400_topk_pareto_swa.pt"
        elif args.stage in RBP400_TRIAGE_STAGES:
            swa_name = "triage_rbp400_triage_pareto_swa.pt"
        else:
            swa_name = "triage_l1_rbp400_pareto_swa.pt"
        swa_ckpt_path = os.path.join(out_dir, swa_name)
        save_checkpoint(swa_ckpt_path, model, cfg, args.stage, -1, swa_score)
        print(f"[pareto-swa] sidecar saved={swa_ckpt_path} metric={swa_score:.6f}", flush=True)
        swa_test_stats = collect_epoch_stats(
            model,
            iter_stage_batches(args.stage, struct_test_loader, pair_test_loader, cfg),
            cfg,
            device,
            -1,
            train_mode=False,
            max_steps=0,
            progress_label=f"{args.stage}/pareto_swa_test",
            progress_every=args.progress_every,
        )
        print(
            f"[pareto-swa][test] {format_rbp400_stage_metrics(args.stage, swa_test_stats, raw_digits=4)}",
            flush=True,
        )
        if swa_score > best_metric + args.min_delta:
            best_metric = swa_score
            best_epoch = -1
            save_checkpoint(ckpt_path, model, cfg, args.stage, best_epoch, best_metric)
            print(f"[pareto-swa] saved averaged checkpoint metric={best_metric:.6f}", flush=True)
        model.load_state_dict(current_state, strict=True)

    print(f"[best] {ckpt_path} metric={best_metric:.6f} epoch={best_epoch}", flush=True)
    if os.path.exists(ckpt_path):
        load_resume(model, ckpt_path, device)
        test_stats = collect_epoch_stats(
            model,
            iter_stage_batches(args.stage, struct_test_loader, pair_test_loader, cfg),
            cfg,
            device,
            best_epoch,
            train_mode=False,
            max_steps=0,
            progress_label=f"{args.stage}/test",
            progress_every=args.progress_every,
        )
        if args.stage in RBP400_STAGES:
            test_text = format_rbp400_stage_metrics(args.stage, test_stats, raw_digits=4)
        elif args.stage in DEST_TRIAGE_STAGES:
            test_text = (
                f"residue_auprc={test_stats['l1_auprc']:.4f} residue_auroc={test_stats['l1_auroc']:.4f} "
                f"residue_mcc={test_stats['l1_mcc']:.4f} acc={test_stats['l1_acc']:.4f} "
                f"precision={test_stats['l1_precision']:.4f} recall={test_stats['l1_recall']:.4f} "
                f"f1={test_stats['l1_f1']:.4f} R@L/5={test_stats['l1_recall_l5']:.4f} "
                f"R@L/10={test_stats['l1_recall_l10']:.4f} P@10={test_stats['l1_precision_10']:.4f} "
                f"Hit@20={test_stats['l1_hit_20']:.4f} Hit@2={test_stats['l1_hit_2']:.4f}"
            )
        elif args.stage in L1_STAGES:
            test_text = (
                f"l1_auprc={test_stats['l1_auprc']:.4f} l1_auroc={test_stats['l1_auroc']:.4f} "
                f"l1_mcc={test_stats['l1_mcc']:.4f} acc={test_stats['l1_acc']:.4f} "
                f"precision={test_stats['l1_precision']:.4f} recall={test_stats['l1_recall']:.4f} "
                f"f1={test_stats['l1_f1']:.4f} R@L/5={test_stats['l1_recall_l5']:.4f} "
                f"R@L/10={test_stats['l1_recall_l10']:.4f} P@10={test_stats['l1_precision_10']:.4f} "
                f"Hit@20={test_stats['l1_hit_20']:.4f} Hit@2={test_stats['l1_hit_2']:.4f}"
            )
        elif args.stage in ("debug", "struct_pretrain"):
            test_text = (
                f"l1={test_stats['l1_auprc']:.4f} l2TopM={test_stats['l2_topm_precision']:.4f} "
                f"l2TopL={test_stats['l2_topl_precision']:.4f}"
            )
        else:
            test_text = (
                f"auprc={test_stats['pair_auprc']:.4f} auroc={test_stats['pair_auroc']:.4f} "
                f"mcc={test_stats['pair_mcc']:.4f} acc={test_stats['acc']:.4f} "
                f"precision={test_stats['precision']:.4f} recall={test_stats['recall']:.4f} "
                f"f1={test_stats['f1']:.4f}"
            )
        print(f"[test] stage={args.stage} best_epoch={best_epoch} {test_text}", flush=True)
        if args.stage in RBP400_STAGES:
            for name, path in rbp400_sidecar_paths.items():
                if not os.path.exists(path):
                    continue
                side_metric, side_epoch = read_checkpoint_score(path, device)
                side_model = TRIAGEPPIModel(cfg).to(device)
                side_missing, _, side_skipped = load_resume(side_model, path, device)
                if side_missing or side_skipped:
                    print(
                        f"[sidecar][skip] name={name} incompatible checkpoint "
                        f"missing={len(side_missing)} skipped_shape={len(side_skipped)} path={path}",
                        flush=True,
                    )
                    del side_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                side_stats = collect_epoch_stats(
                    side_model,
                    iter_stage_batches(args.stage, struct_test_loader, pair_test_loader, cfg),
                    cfg,
                    device,
                    side_epoch,
                    train_mode=False,
                    max_steps=0,
                    progress_label=f"{args.stage}/sidecar_{name}_test",
                    progress_every=args.progress_every,
                )
                print(
                    f"[sidecar][test] name={name} epoch={side_epoch} val_metric={side_metric:.6f} "
                    f"{format_rbp400_stage_metrics(args.stage, side_stats, raw_digits=4)}",
                    flush=True,
                )
                del side_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if args.stage == "l1_rbp400_topk":
                swa_name = "triage_l1_rbp400_topk_pareto_swa.pt"
            elif args.stage in RBP400_TRIAGE_STAGES:
                swa_name = "triage_rbp400_triage_pareto_swa.pt"
            else:
                swa_name = "triage_l1_rbp400_pareto_swa.pt"
            swa_ckpt_path = os.path.join(out_dir, swa_name)
            if os.path.exists(swa_ckpt_path):
                best_model = TRIAGEPPIModel(cfg).to(device)
                swa_model = TRIAGEPPIModel(cfg).to(device)
                best_missing, _, best_skipped = load_resume(best_model, ckpt_path, device)
                swa_missing, _, swa_skipped = load_resume(swa_model, swa_ckpt_path, device)
                if best_missing or best_skipped or swa_missing or swa_skipped:
                    print(
                        f"[ensemble][skip] incompatible best/swa checkpoints "
                        f"best_missing={len(best_missing)} best_skipped={len(best_skipped)} "
                        f"swa_missing={len(swa_missing)} swa_skipped={len(swa_skipped)}",
                        flush=True,
                    )
                else:
                    ens_stats = collect_l1_probability_ensemble_stats(
                        [best_model, swa_model],
                        iter_stage_batches(args.stage, struct_test_loader, pair_test_loader, cfg),
                        cfg,
                        device,
                    )
                    print(
                        f"[ensemble][test] best+swa {format_rbp400_stage_metrics(args.stage, ens_stats, raw_digits=4, full_triage=False)}",
                        flush=True,
                    )
    print(f"[metrics] {metrics_path}", flush=True)


def train(args):
    if args.stage == "topk_engineering_final":
        stage_args = argparse.Namespace(**vars(args))
        stage_args.stage = "joint_finetune"
        stage_args.topk_engineering = True
        print("[pipeline] top-k engineering final only: joint_finetune -> triage_topk_final.pt", flush=True)
        train_one_stage(stage_args)
        return

    if args.stage not in ("engineering", "topk_engineering"):
        train_one_stage(args)
        return
    topk_engineering = args.stage == "topk_engineering"
    label = "top-k engineering" if topk_engineering else "engineering"
    print(f"[pipeline] {label} stages: struct_pretrain -> pair_fusion -> joint_finetune", flush=True)
    for stage in ENGINEERING_STAGES:
        if args.skip_existing_struct and (stage == "struct_pretrain" or (topk_engineering and stage == "pair_fusion")):
            ckpt_path = os.path.join(args.out_dir or DEFAULT_OUT_DIR, ckpt_name_for_stage(stage, topk_engineering=topk_engineering))
            if os.path.exists(ckpt_path):
                score, epoch = read_checkpoint_score(ckpt_path, torch.device("cpu"))
                print(f"[pipeline] skip stage={stage} existing={ckpt_path} metric={score:.6f} epoch={epoch}", flush=True)
                continue
        stage_args = argparse.Namespace(**vars(args))
        stage_args.stage = stage
        stage_args.topk_engineering = topk_engineering
        print("\n" + "=" * 90, flush=True)
        print(f"[pipeline] start stage={stage} namespace={label}", flush=True)
        print("=" * 90, flush=True)
        train_one_stage(stage_args)
    print(f"[pipeline] {label} training finished", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="PertiNet training entry point")
    p.add_argument("--stage", choices=["debug", "struct_pretrain", "pair_fusion", "joint_finetune", "l1_dest", "dest_triage", "l1_graphrbf", "l1_rbp400", "l1_rbp400_topk", "rbp400_triage", "tuna_pair_finetune", "engineering", "topk_engineering", "topk_engineering_final"], default="engineering")
    p.add_argument("--resume", default="")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-res-in", type=int, default=0, help="Override config d_res_in. 0 uses triage_config.py.")
    p.add_argument("--d-model", type=int, default=0, help="Override config d_model. 0 uses triage_config.py.")
    p.add_argument("--topk", type=int, default=0, help="Override config topk. 0 uses triage_config.py.")
    p.add_argument("--topm", type=int, default=0, help="Override config topm. 0 uses triage_config.py.")
    p.add_argument("--batch-size", type=int, default=0, help="Override batch size. 0 uses stage/config defaults; L1 defaults to cfg.l1_batch_size.")
    p.add_argument("--l1-batch-size", type=int, default=0, help="Override config L1/GraphRBF-PP batch size. 0 uses triage_config.py.")
    p.add_argument("--l1-lr", type=float, default=0.0, help="Override L1 learning rate. 0 uses triage_config.py.")
    p.add_argument("--l1-pos-weight-cap", type=float, default=0.0, help="Override L1 positive-class weight cap. 0 uses triage_config.py.")
    p.add_argument("--w-l1-rank", type=float, default=-1.0, help="Override L1 pairwise rank loss weight. -1 uses triage_config.py.")
    p.add_argument("--l1-rank-margin", type=float, default=0.0, help="Override L1 pairwise rank margin. 0 uses triage_config.py.")
    p.add_argument("--w-l1-hard-rank", type=float, default=-1.0, help="Override L1 hard-rank loss weight. -1 uses triage_config.py.")
    p.add_argument("--l1-hard-rank-start", type=int, default=0, help="Override hard-rank start epoch. 0 uses triage_config.py.")
    p.add_argument("--l1-hard-rank-ramp", type=int, default=0, help="Override hard-rank ramp epochs. 0 uses triage_config.py.")
    p.add_argument("--w-l1-topband-bce", type=float, default=-1.0, help="Override top-band BCE loss weight. -1 uses triage_config.py.")
    p.add_argument("--l1-topband-start", type=int, default=0, help="Override top-band BCE start epoch. 0 uses triage_config.py.")
    p.add_argument("--l1-topband-ramp", type=int, default=0, help="Override top-band BCE ramp epochs. 0 uses triage_config.py.")
    p.add_argument("--l1-topband-frac", type=float, default=0.0, help="Override top-band fraction. 0 uses triage_config.py.")
    p.add_argument("--w-l1-l10-boundary", type=float, default=-1.0, help="Override L/10 boundary recall loss weight. -1 uses triage_config.py.")
    p.add_argument("--l1-l10-boundary-start", type=int, default=0, help="Override L/10 boundary loss start epoch. 0 uses triage_config.py.")
    p.add_argument("--l1-l10-boundary-ramp", type=int, default=0, help="Override L/10 boundary loss ramp epochs. 0 uses triage_config.py.")
    p.add_argument("--l1-l10-boundary-margin", type=float, default=0.0, help="Override L/10 boundary loss margin. 0 uses triage_config.py.")
    p.add_argument("--l1-extreme-label-weight", type=float, default=0.0, help="Override zero/full-label protein loss weight. 0 uses triage_config.py.")
    p.add_argument("--l1-zero-label-weight", type=float, default=-1.0, help="Override zero-label protein loss weight. -1 uses triage_config.py.")
    p.add_argument("--l1-full-label-weight", type=float, default=-1.0, help="Override full-label protein loss weight. -1 uses triage_config.py.")
    p.add_argument("--include-zero-label-proteins", action="store_true", help="For RBP400 ablation, keep all-zero label proteins in L1 splits.")
    p.add_argument("--l1-ager-alpha", type=float, default=-1.0, help="Override RBP400 AGER alpha. -1 uses triage_config.py.")
    p.add_argument("--l1-ager-radius", type=float, default=0.0, help="Override RBP400 AGER radius. 0 uses triage_config.py.")
    p.add_argument("--l1-ager-top-m", type=int, default=0, help="Override RBP400 AGER top-m. 0 uses triage_config.py.")
    p.add_argument("--num-workers", type=int, default=-1, help="Override config DataLoader workers. -1 uses triage_config.py.")
    p.add_argument("--l1-num-workers", type=int, default=-1, help="Override config L1 DataLoader workers. -1 uses triage_config.py.")
    p.add_argument("--val-fraction", type=float, default=0.0, help="Override config validation fraction. 0 uses triage_config.py.")
    p.add_argument("--test-fraction", type=float, default=0.0, help="Override config test fraction. 0 uses triage_config.py.")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--debug-samples", type=int, default=4)
    p.add_argument("--max-items", type=int, default=0, help="Optional cap for real datasets; 0 means all.")
    p.add_argument("--pp-root", default="", help="Override GraphRBF-PP prepared root for --stage l1_graphrbf.")
    p.add_argument("--rbp400-root", default="", help="Override RBP400 prepared root for RBP400 L1 stages.")
    p.add_argument("--rbp400-id-list", default="", help="Override RBP400 accession list used for automatic 8:1:1 split.")
    p.add_argument("--tuna-train-fourpack-dir", default="", help="Fixed TUnA Intra1 training fourpack directory.")
    p.add_argument("--tuna-val-fourpack-dir", default="", help="Fixed TUnA Intra0 validation fourpack directory.")
    p.add_argument("--tuna-test-fourpack-dir", default="", help="Fixed TUnA Intra2 test fourpack directory.")
    p.add_argument("--max-pair-len", type=int, default=0, help="Override config max_pair_len. 0 uses triage_config.py.")
    p.add_argument("--max-site-len", type=int, default=0, help="Override config max_site_len. 0 uses triage_config.py.")
    p.add_argument("--train-max-steps", type=int, default=-1, help="-1 uses engineering defaults, 0 uses a full epoch, positive caps train batches.")
    p.add_argument("--val-max-steps", type=int, default=-1, help="-1 uses engineering defaults, 0 uses full validation, positive caps validation batches.")
    p.add_argument("--full-epoch", action="store_true", help="Disable engineering step caps and iterate the full dataset each epoch.")
    p.add_argument("--progress-every", type=int, default=100, help="Print intra-epoch progress every N batches; 0 disables.")
    p.add_argument("--split-mode", choices=["random", "group", "protein_pair", "protein_component", "protein_disjoint"], default="protein_disjoint", help="Train/val/test split policy for real datasets.")
    p.add_argument("--balance-pair-train", action="store_true", help="Downsample positive TUnA pair training examples after splitting; validation/test stay unchanged.")
    p.add_argument("--balance-pair-ratio", type=float, default=1.0, help="Positive:negative ratio used with --balance-pair-train.")
    p.add_argument("--patience", type=int, default=-1, help="Override early stop patience. -1 uses config default; 0 disables.")
    p.add_argument("--min-delta", type=float, default=1e-6, help="Minimum validation-score improvement required to overwrite the best checkpoint.")
    p.add_argument("--reset-best", action="store_true", help="Ignore the existing target checkpoint score and allow this run to replace it.")
    p.add_argument("--freeze-pair-local-epochs", type=int, default=0, help="During pair_fusion, freeze L1/L2/local encoder modules for the first N epochs.")
    p.add_argument("--use-scheduler", action="store_true", default=True, help="Use warmup + cosine LR schedule instead of constant LR.")
    p.add_argument("--no-scheduler", dest="use_scheduler", action="store_false", help="Disable warmup + cosine LR schedule and use constant LR.")
    p.add_argument("--score-loss-penalty", type=float, default=0.0, help="Optional loss penalty for pair/joint checkpoint score; default preserves old score scale.")
    p.add_argument("--skip-existing-struct", action="store_true", default=True, help="In engineering pipeline, skip struct_pretrain when triage_stage1_struct.pt already exists.")
    p.add_argument("--rerun-struct", dest="skip_existing_struct", action="store_false", help="Force engineering pipeline to rerun struct_pretrain.")
    p.add_argument("--auto-esm", action="store_true", default=True, help="Before training, compute missing ESM .npy cache files and reuse them later.")
    p.add_argument("--no-auto-esm", dest="auto_esm", action="store_false", help="Do not compute ESM cache before training.")
    p.add_argument("--auto-site-fasta", action="store_true", default=True, help="Build site_data/site_global/proteins.fasta from DSSP when missing.")
    p.add_argument("--no-auto-site-fasta", dest="auto_site_fasta", action="store_false", help="Do not auto-build SITE FASTA.")
    p.add_argument("--esm-model", default=DEFAULT_ESM_MODEL, help="Local esm2_t33_650M_UR50D.pt path used by scripts/compute_esm.py.")
    p.add_argument("--esm-batch", type=int, default=8, help="Batch size for automatic ESM precomputation.")
    p.add_argument("--esm-max-len", type=int, default=1022, help="Maximum sequence length for automatic ESM precomputation.")
    p.add_argument("--epochs", type=int, default=0, help="Override stage epochs. 0 uses config defaults.")
    p.add_argument("--synthetic-debug", action="store_true", help="Use synthetic toy data instead of repaired real data.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
