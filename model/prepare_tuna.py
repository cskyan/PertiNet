#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and stage one supplied TUnA level for PertiNet.

Default input:
  data/TUnA/Intra0

Default output:
  outputs/TUnA/Intra0

The script checks:
  - expected directory/files used by current TRIAGE pair training
  - pair table and fourpack consistency
  - sequence coverage and pair UID coverage
  - PSSM numeric validity, all-zero rate, length match
  - DSSP SS/RSA/ASA validity, all-zero rate, length match
  - PSSM/DSSP/structure fail-list counts

It also copies the current model-required content into:
  <out_root>/prepared/{seq,pssm,dssp,struct,fourpack}
and writes audit reports into:
  <out_root>/audit
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data/TUnA/Intra0")
DEFAULT_OUT = Path("outputs/TUnA/Intra0")
AA20 = list("ARNDCQEGHILKMFPSTWYV")
SS8 = set(["H", "G", "I", "E", "B", "T", "S", "C"])
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}
EXPECTED_DIRS = ("seq", "pssm", "dssp", "struct", "fourpack")
OPTIONAL_DIRS = ("coords",)
FOURPACK_FILES = (
    "tuna.seq.fasta",
    "tuna.all.pssm.tsv",
    "tuna.all.dssp.tsv",
    "tuna.shapes.tsv",
    "tuna.contacts.tsv",
)
ROOT_TABLES = ("pairs.tsv", "shapes.tsv")
FAIL_FILES = (
    "_fail_af.txt",
    "_fail_dssp.txt",
    "_pssm.fail.txt",
    "_refetch_fail_af.txt",
    "_refetch_fail_dssp.txt",
)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    first = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                first = line.strip()
                break
    if first.startswith("#") and "\t" not in first:
        header = first.lstrip("#").strip().split()
        return pd.read_csv(path, sep=r"\s+", comment="#", names=header, engine="python")
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        return pd.read_csv(path, sep=r"\s+", engine="python")
    return df


def parse_fasta_lengths(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    cur = None
    seq: List[str] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
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
                seq.append("".join(ch for ch in line.upper() if ch.isalpha()))
        if cur is not None:
            out[cur] = len("".join(seq))
    return out


def parse_fasta_sequences(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    cur = None
    seq: List[str] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    out[cur] = "".join(seq)
                cur = line[1:].split()[0]
                seq = []
            else:
                seq.append("".join(ch for ch in line.upper() if ch.isalpha()))
        if cur is not None:
            out[cur] = "".join(seq)
    return out


def first_col(df: pd.DataFrame, names: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def uid_col(df: pd.DataFrame) -> str:
    return first_col(df, ["uid", "chain_uid", "protein", "protein_id", "id"], df.columns[0]) or df.columns[0]


def numeric_cols(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    skip = {str(x).lower() for x in exclude}
    cols: List[str] = []
    for c in df.columns:
        if str(c).lower() in skip:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() > 0:
            cols.append(c)
    return cols


def pssm_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in AA20 if c in df.columns]
    if len(cols) == 20:
        return cols
    ucol = uid_col(df)
    return numeric_cols(df, [ucol, "idx", "pos", "aa", "residue", "res"])[:20]


def dssp_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {str(c).lower(): c for c in df.columns}
    return {
        "uid": lower.get("chain_uid") or lower.get("uid") or lower.get("protein") or lower.get("protein_id") or df.columns[0],
        "idx": lower.get("idx") or lower.get("pos") or lower.get("res_idx") or lower.get("residue_index"),
        "aa": lower.get("aa") or lower.get("residue") or lower.get("res"),
        "ss": lower.get("ss") or lower.get("sec") or lower.get("secondary_structure"),
        "asa": lower.get("asa") or lower.get("acc") or lower.get("solvent_accessibility"),
        "rsa": lower.get("rsa") or lower.get("rasa") or lower.get("rel_asa"),
    }


def infer_chain_id(uid: str) -> str:
    token = str(uid)
    for sep in ("_", ":", "-", "."):
        if sep in token:
            tail = token.rsplit(sep, 1)[-1]
            if 1 <= len(tail) <= 3:
                return tail
    return ""


def parse_pdb_ca(path: Path, chain_hint: str = "") -> Optional[np.ndarray]:
    coords = []
    seen = set()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain = line[21].strip()
                if chain_hint and chain and chain != chain_hint:
                    continue
                key = (chain, line[22:27].strip(), line[27].strip())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    except OSError:
        return None
    return np.asarray(coords, dtype=np.float32) if coords else None


def parse_mmcif_ca(path: Path, chain_hint: str = "") -> Optional[np.ndarray]:
    coords = []
    seen = set()
    in_loop = False
    cols: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line == "loop_":
                    in_loop = True
                    cols = []
                    continue
                if in_loop and line.startswith("_atom_site."):
                    cols.append(line.split()[0])
                    continue
                if in_loop and cols and not line.startswith("_"):
                    parts = line.split()
                    if len(parts) < len(cols):
                        continue
                    row = {c: parts[i] for i, c in enumerate(cols)}
                    atom = row.get("_atom_site.label_atom_id") or row.get("_atom_site.auth_atom_id") or ""
                    if atom.strip("\"'") != "CA":
                        continue
                    chain = (
                        row.get("_atom_site.auth_asym_id")
                        or row.get("_atom_site.label_asym_id")
                        or ""
                    ).strip("\"'")
                    if chain_hint and chain and chain != chain_hint:
                        continue
                    res_id = (
                        chain,
                        (row.get("_atom_site.auth_seq_id") or row.get("_atom_site.label_seq_id") or "").strip("\"'"),
                        (row.get("_atom_site.pdbx_PDB_ins_code") or "").strip("\"'"),
                    )
                    if res_id in seen:
                        continue
                    seen.add(res_id)
                    try:
                        coords.append((
                            float((row.get("_atom_site.Cartn_x") or "nan").strip("\"'")),
                            float((row.get("_atom_site.Cartn_y") or "nan").strip("\"'")),
                            float((row.get("_atom_site.Cartn_z") or "nan").strip("\"'")),
                        ))
                    except ValueError:
                        continue
    except OSError:
        return None
    return np.asarray(coords, dtype=np.float32) if coords else None


def find_structure_file(struct_dir: Path, uid: str) -> Optional[Path]:
    if not struct_dir.exists():
        return None
    candidates = [
        uid,
        uid.lower(),
        uid.upper(),
        uid.replace(":", "_"),
        uid.replace("-", "_"),
    ]
    exts = (".pdb", ".ent", ".cif", ".mmcif")
    for stem in candidates:
        for ext in exts:
            p = struct_dir / f"{stem}{ext}"
            if p.exists():
                return p
    uid_l = uid.lower()
    for p in struct_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and uid_l in p.stem.lower():
            return p
    return None


def load_coords_for_uid(struct_dir: Path, uid: str) -> Optional[np.ndarray]:
    path = find_structure_file(struct_dir, uid)
    if path is None:
        return None
    chain_hint = infer_chain_id(uid)
    if path.suffix.lower() in (".cif", ".mmcif"):
        arr = parse_mmcif_ca(path, chain_hint)
        if arr is None and chain_hint:
            arr = parse_mmcif_ca(path, "")
        return arr
    arr = parse_pdb_ca(path, chain_hint)
    if arr is None and chain_hint:
        arr = parse_pdb_ca(path, "")
    return arr


def rsa_from_coords(coords: np.ndarray, L: int) -> Optional[np.ndarray]:
    c = np.asarray(coords, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] < 3 or c.shape[0] != L:
        return None
    c = c[:, :3]
    finite = np.isfinite(c).all(axis=1)
    if not finite.any():
        return None
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    valid = finite[:, None] & finite[None, :]
    np.fill_diagonal(valid, False)
    density10 = ((dist < 10.0) & valid).sum(axis=1).astype(np.float32)
    density14 = ((dist < 14.0) & valid).sum(axis=1).astype(np.float32)
    rsa = 1.0 / (1.0 + 0.08 * density10 + 0.03 * density14)
    return np.clip(rsa, 0.02, 1.0).astype(np.float32)


def fallback_rsa_from_ss_len(ss: Sequence[str], L: int) -> np.ndarray:
    ss = [str(x).strip().upper() for x in ss]
    rsa = np.full(L, 0.42, dtype=np.float32)
    for i, s in enumerate(ss[:L]):
        if s in ("H", "G", "I"):
            rsa[i] = 0.34
        elif s in ("E", "B"):
            rsa[i] = 0.30
        elif s in ("T", "S"):
            rsa[i] = 0.52
        else:
            rsa[i] = 0.58
    if L > 1:
        pos = np.linspace(0.0, 1.0, L, dtype=np.float32)
        terminal = np.maximum(0.0, 1.0 - np.minimum(pos, 1.0 - pos) / 0.20)
        rsa = np.clip(rsa + 0.15 * terminal, 0.05, 0.95)
    return rsa.astype(np.float32)


def repair_dssp_table(
    src_path: Path,
    fasta_path: Path,
    struct_dir: Path,
    out_path: Path,
    coords_dir: Path,
    report_path: Path,
) -> Dict:
    if not src_path.exists():
        return {"enabled": False, "reason": "missing_dssp_table"}
    df = read_table(src_path)
    cols = dssp_cols(df)
    ucol = cols["uid"] or df.columns[0]
    ss_col = cols["ss"]
    rsa_col = cols["rsa"]
    asa_col = cols["asa"]
    if rsa_col is None:
        rsa_col = "rsa"
        df[rsa_col] = 0.0
    if asa_col is None:
        asa_col = "asa"
        df[asa_col] = 0.0
    seqs = parse_fasta_sequences(fasta_path)
    coords_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    repaired_frames = []
    coord_ok = 0
    coord_len_match = 0
    repaired_uid = 0
    for uid, sub in df.groupby(ucol, sort=False):
        uid_s = str(uid)
        sub = sub.copy()
        L = len(sub)
        current = pd.to_numeric(sub[rsa_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        needs_repair = bool(np.abs(current).sum() == 0)
        coords = load_coords_for_uid(struct_dir, uid_s)
        coord_source = "none"
        if coords is not None and coords.ndim == 2 and coords.shape[1] >= 3:
            coord_ok += 1
            coord_source = "struct"
            np.save(coords_dir / f"{uid_s}.npy", coords[:, :3].astype(np.float32))
            if coords.shape[0] == L:
                coord_len_match += 1
        new_rsa = current
        repair_source = "original"
        if needs_repair:
            repaired_uid += 1
            coord_rsa = rsa_from_coords(coords, L) if coords is not None else None
            if coord_rsa is not None:
                new_rsa = coord_rsa
                repair_source = "coords"
            else:
                if ss_col:
                    ss = sub[ss_col].astype(str).tolist()
                else:
                    ss = ["C"] * L
                new_rsa = fallback_rsa_from_ss_len(ss, L)
                repair_source = "ss_len_fallback"
            sub[rsa_col] = new_rsa
            sub[asa_col] = np.asarray(new_rsa, dtype=np.float32) * 200.0
        repaired_frames.append(sub)
        rows.append({
            "uid": uid_s,
            "dssp_len": L,
            "seq_len": len(seqs.get(uid_s, "")) if uid_s in seqs else "",
            "coord_source": coord_source,
            "coord_len": int(coords.shape[0]) if coords is not None else 0,
            "coord_len_match_dssp": bool(coords is not None and coords.shape[0] == L),
            "rsa_repaired": needs_repair,
            "repair_source": repair_source,
            "rsa_mean_after": float(np.asarray(new_rsa).mean()) if len(new_rsa) else 0.0,
            "rsa_all_zero_after": bool(np.abs(np.asarray(new_rsa)).sum() == 0),
        })
    out_df = pd.concat(repaired_frames, ignore_index=True) if repaired_frames else df
    out_df.to_csv(out_path, sep="\t", index=False)
    pd.DataFrame(rows).to_csv(report_path, sep="\t", index=False)
    after = pd.to_numeric(out_df[rsa_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return {
        "enabled": True,
        "out_path": str(out_path),
        "n_uid": int(len(rows)),
        "n_uid_repaired_rsa": int(repaired_uid),
        "n_uid_with_coords": int(coord_ok),
        "n_uid_coord_len_match": int(coord_len_match),
        "rsa_all_zero_after": bool(np.abs(after).sum() == 0),
        "rsa_mean_after": float(np.mean(after)) if after.size else 0.0,
        "report_path": str(report_path),
        "coords_dir": str(coords_dir),
    }


def summarize_pssm(path: Path, seq_lens: Dict[str, int]) -> Tuple[pd.DataFrame, Dict]:
    if not path.exists():
        return pd.DataFrame(), {"exists": False, "path": str(path)}
    df = read_table(path)
    ucol = uid_col(df)
    vcols = pssm_cols(df)
    rows = []
    if len(vcols) < 20:
        return pd.DataFrame(), {"exists": True, "path": str(path), "n_value_cols": len(vcols), "error": "less_than_20_numeric_columns"}
    for uid, sub in df.groupby(ucol, sort=False):
        uid_s = str(uid)
        mat = sub[vcols[:20]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(mat)
        mat0 = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        exp_len = seq_lens.get(uid_s)
        rows.append({
            "uid": uid_s,
            "pssm_len": int(len(sub)),
            "seq_len": int(exp_len) if exp_len is not None else "",
            "len_match_seq": bool(exp_len == len(sub)) if exp_len is not None else "",
            "nan_count": int(np.isnan(mat).sum()),
            "inf_count": int(np.isinf(mat).sum()),
            "finite_rate": float(finite.mean()) if finite.size else 0.0,
            "all_zero": bool(np.abs(mat0).sum() == 0),
            "near_zero": bool(np.abs(mat0).mean() < 1e-8) if mat0.size else True,
            "abs_mean": float(np.abs(mat0).mean()) if mat0.size else 0.0,
            "abs_max": float(np.abs(mat0).max()) if mat0.size else 0.0,
            "nonzero_rate": float((mat0 != 0).mean()) if mat0.size else 0.0,
        })
    by_uid = pd.DataFrame(rows)
    with_len = by_uid[by_uid["seq_len"].astype(str) != ""] if len(by_uid) else by_uid
    summary = {
        "exists": True,
        "path": str(path),
        "n_uid": int(len(by_uid)),
        "n_rows": int(len(df)),
        "n_value_cols": int(len(vcols[:20])),
        "len_match_seq_rate": float(with_len["len_match_seq"].mean()) if len(with_len) else 0.0,
        "all_zero_uid_rate": float(by_uid["all_zero"].mean()) if len(by_uid) else 0.0,
        "near_zero_uid_rate": float(by_uid["near_zero"].mean()) if len(by_uid) else 0.0,
        "mean_abs_mean": float(by_uid["abs_mean"].mean()) if len(by_uid) else 0.0,
        "mean_nonzero_rate": float(by_uid["nonzero_rate"].mean()) if len(by_uid) else 0.0,
        "nan_uid_count": int((by_uid["nan_count"] > 0).sum()) if len(by_uid) else 0,
        "inf_uid_count": int((by_uid["inf_count"] > 0).sum()) if len(by_uid) else 0,
    }
    return by_uid, summary


def summarize_dssp(path: Path, seq_lens: Dict[str, int]) -> Tuple[pd.DataFrame, Dict]:
    if not path.exists():
        return pd.DataFrame(), {"exists": False, "path": str(path)}
    df = read_table(path)
    cols = dssp_cols(df)
    ucol = cols["uid"] or df.columns[0]
    ss_col = cols["ss"]
    rsa_col = cols["rsa"]
    asa_col = cols["asa"]
    idx_col = cols["idx"]
    rows = []
    for uid, sub in df.groupby(ucol, sort=False):
        uid_s = str(uid)
        exp_len = seq_lens.get(uid_s)
        if ss_col:
            ss = sub[ss_col].astype(str).str.strip().str.upper()
            ss_valid_rate = float(ss.isin(SS8).mean()) if len(ss) else 0.0
            ss_noncoil_rate = float((~ss.isin(["", "C", "L", "-", "?", "X", "NAN", "NONE"])).mean()) if len(ss) else 0.0
        else:
            ss_valid_rate = 0.0
            ss_noncoil_rate = 0.0
        rsa = pd.to_numeric(sub[rsa_col], errors="coerce").to_numpy(dtype=np.float64) if rsa_col else np.zeros(len(sub), dtype=np.float64)
        asa = pd.to_numeric(sub[asa_col], errors="coerce").to_numpy(dtype=np.float64) if asa_col else np.zeros(len(sub), dtype=np.float64)
        rsa0 = np.nan_to_num(rsa, nan=0.0, posinf=0.0, neginf=0.0)
        asa0 = np.nan_to_num(asa, nan=0.0, posinf=0.0, neginf=0.0)
        idx_cont = ""
        if idx_col:
            idx = pd.to_numeric(sub[idx_col], errors="coerce").dropna().to_numpy(dtype=np.int64)
            idx_cont = bool(idx.size == len(sub) and idx.min() == 1 and idx.max() == len(sub) and len(np.unique(idx)) == len(sub)) if idx.size else False
        rows.append({
            "uid": uid_s,
            "dssp_len": int(len(sub)),
            "seq_len": int(exp_len) if exp_len is not None else "",
            "len_match_seq": bool(exp_len == len(sub)) if exp_len is not None else "",
            "idx_continuous_1_to_len": idx_cont,
            "ss_valid_rate": ss_valid_rate,
            "ss_noncoil_rate": ss_noncoil_rate,
            "rsa_nan_count": int(np.isnan(rsa).sum()) if rsa_col else 0,
            "asa_nan_count": int(np.isnan(asa).sum()) if asa_col else 0,
            "rsa_all_zero": bool(np.abs(rsa0).sum() == 0),
            "asa_all_zero": bool(np.abs(asa0).sum() == 0),
            "rsa_mean": float(rsa0.mean()) if rsa0.size else 0.0,
            "asa_mean": float(asa0.mean()) if asa0.size else 0.0,
        })
    by_uid = pd.DataFrame(rows)
    with_len = by_uid[by_uid["seq_len"].astype(str) != ""] if len(by_uid) else by_uid
    summary = {
        "exists": True,
        "path": str(path),
        "n_uid": int(len(by_uid)),
        "n_rows": int(len(df)),
        "has_ss_col": bool(ss_col),
        "has_rsa_col": bool(rsa_col),
        "has_asa_col": bool(asa_col),
        "len_match_seq_rate": float(with_len["len_match_seq"].mean()) if len(with_len) else 0.0,
        "mean_ss_valid_rate": float(by_uid["ss_valid_rate"].mean()) if len(by_uid) else 0.0,
        "mean_ss_noncoil_rate": float(by_uid["ss_noncoil_rate"].mean()) if len(by_uid) else 0.0,
        "rsa_all_zero_uid_rate": float(by_uid["rsa_all_zero"].mean()) if len(by_uid) else 0.0,
        "asa_all_zero_uid_rate": float(by_uid["asa_all_zero"].mean()) if len(by_uid) else 0.0,
        "mean_rsa_mean": float(by_uid["rsa_mean"].mean()) if len(by_uid) else 0.0,
        "mean_asa_mean": float(by_uid["asa_mean"].mean()) if len(by_uid) else 0.0,
        "nan_uid_count": int(((by_uid["rsa_nan_count"] > 0) | (by_uid["asa_nan_count"] > 0)).sum()) if len(by_uid) else 0,
    }
    return by_uid, summary


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def choose_pair_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lower = {str(c).lower(): c for c in df.columns}
    pairs = [
        ("a_uid", "b_uid"),
        ("uid_a", "uid_b"),
        ("auid", "buid"),
        ("uid1", "uid2"),
        ("protein_a", "protein_b"),
        ("chain_a", "chain_b"),
        ("protein_a", "protein_b"),
        ("a", "b"),
        ("prot_a", "prot_b"),
        ("p1", "p2"),
    ]
    for a, b in pairs:
        if a in lower and b in lower:
            return lower[a], lower[b]
    if len(df.columns) >= 2:
        return df.columns[0], df.columns[1]
    return None, None


def pair_uid_set(path: Path) -> Tuple[set, int, Dict]:
    if not path.exists():
        return set(), 0, {"exists": False, "path": str(path)}
    df = read_table(path)
    a_col, b_col = choose_pair_cols(df)
    if not a_col or not b_col:
        return set(), len(df), {"exists": True, "path": str(path), "n_rows": len(df), "error": "no_pair_columns"}
    a = df[a_col].astype(str).tolist()
    b = df[b_col].astype(str).tolist()
    uids = set(a) | set(b)
    summary = {
        "exists": True,
        "path": str(path),
        "n_rows": int(len(df)),
        "a_col": str(a_col),
        "b_col": str(b_col),
        "n_unique_pair_uid": int(len(uids)),
    }
    return uids, len(df), summary


def write_summary_tsv(summary: Dict, path: Path):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        for k in sorted(summary):
            writer.writerow([k, summary[k]])


def copy_key_files(src_root: Path, out_root: Path, copy_required_dirs: bool = True):
    prepared = out_root / "prepared"
    fourpack_out = prepared / "fourpack"
    fourpack_out.mkdir(parents=True, exist_ok=True)
    for name in FOURPACK_FILES:
        src = src_root / "fourpack" / name
        if src.exists():
            shutil.copy2(src, fourpack_out / name)
    for name in ROOT_TABLES:
        src = src_root / name
        if src.exists():
            shutil.copy2(src, prepared / name)
    if copy_required_dirs:
        for d in EXPECTED_DIRS:
            src = src_root / d
            dst = prepared / d
            if src.exists() and src.is_dir() and src.resolve() != dst.resolve():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="Root of one original TUnA level.")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT), help="Output root under the project workspace.")
    ap.add_argument("--no-copy-data", action="store_true", help="Only write audit reports and key tables; do not copy seq/pssm/dssp/struct/fourpack directories.")
    ap.add_argument("--no-repair-dssp", action="store_true", help="Do not repair prepared/fourpack/tuna.all.dssp.tsv RSA/ASA values.")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    audit = out_root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise FileNotFoundError(f"TUnA level root not found: {root}")

    dir_summary = {}
    for d in EXPECTED_DIRS:
        p = root / d
        dir_summary[f"has_dir_{d}"] = p.exists() and p.is_dir()
        dir_summary[f"n_files_{d}"] = sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0
    for d in OPTIONAL_DIRS:
        p = root / d
        dir_summary[f"has_optional_dir_{d}"] = p.exists() and p.is_dir()
        dir_summary[f"n_files_optional_{d}"] = sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0

    fourpack = root / "fourpack"
    file_summary = {}
    for name in FOURPACK_FILES:
        p = fourpack / name
        file_summary[f"has_fourpack_{name}"] = p.exists()
        file_summary[f"size_fourpack_{name}"] = p.stat().st_size if p.exists() else 0
    for name in ROOT_TABLES:
        p = root / name
        file_summary[f"has_root_{name}"] = p.exists()
        file_summary[f"size_root_{name}"] = p.stat().st_size if p.exists() else 0

    fail_summary = {}
    for name in FAIL_FILES:
        p = root / name
        fail_summary[f"has_{name}"] = p.exists()
        fail_summary[f"n_{name}"] = count_lines(p)

    seq_lens = parse_fasta_lengths(fourpack / "tuna.seq.fasta")
    seq_summary = {
        "seq_n_uid": len(seq_lens),
        "seq_n_residue": int(sum(seq_lens.values())) if seq_lens else 0,
        "seq_min_len": int(min(seq_lens.values())) if seq_lens else 0,
        "seq_max_len": int(max(seq_lens.values())) if seq_lens else 0,
        "seq_mean_len": float(np.mean(list(seq_lens.values()))) if seq_lens else 0.0,
    }

    pssm_by_uid, pssm_summary = summarize_pssm(fourpack / "tuna.all.pssm.tsv", seq_lens)
    dssp_by_uid, dssp_summary = summarize_dssp(fourpack / "tuna.all.dssp.tsv", seq_lens)
    if len(pssm_by_uid):
        pssm_by_uid.to_csv(audit / "tuna_pssm_quality_by_uid.tsv", sep="\t", index=False)
    if len(dssp_by_uid):
        dssp_by_uid.to_csv(audit / "tuna_dssp_quality_by_uid.tsv", sep="\t", index=False)

    shape_uids, n_shapes, shapes_summary = pair_uid_set(fourpack / "tuna.shapes.tsv")
    root_pair_uids, n_pairs, pairs_summary = pair_uid_set(root / "pairs.tsv")
    contacts_path = fourpack / "tuna.contacts.tsv"
    contacts_summary = {"exists": contacts_path.exists(), "path": str(contacts_path)}
    if contacts_path.exists():
        try:
            contacts = read_table(contacts_path)
            contacts_summary["n_rows"] = int(len(contacts))
            contacts_summary["n_cols"] = int(len(contacts.columns))
        except Exception as exc:
            contacts_summary["error"] = repr(exc)

    expected_uids = set(seq_lens)
    expected_uids |= shape_uids
    expected_uids |= root_pair_uids
    pssm_uids = set(pssm_by_uid["uid"].astype(str)) if len(pssm_by_uid) else set()
    dssp_uids = set(dssp_by_uid["uid"].astype(str)) if len(dssp_by_uid) else set()
    uid_rows = []
    for uid in sorted(expected_uids):
        uid_rows.append({
            "uid": uid,
            "in_seq": uid in seq_lens,
            "seq_len": seq_lens.get(uid, ""),
            "in_pair_shapes": uid in shape_uids,
            "in_root_pairs": uid in root_pair_uids,
            "has_pssm": uid in pssm_uids,
            "has_dssp": uid in dssp_uids,
        })
    uid_audit = pd.DataFrame(uid_rows)
    uid_audit.to_csv(audit / "tuna_uid_coverage.tsv", sep="\t", index=False)

    n_uid = max(len(expected_uids), 1)
    coverage_summary = {
        "n_expected_uid": int(len(expected_uids)),
        "n_pair_shapes": int(n_shapes),
        "n_root_pairs": int(n_pairs),
        "uid_seq_rate": float(uid_audit["in_seq"].mean()) if len(uid_audit) else 0.0,
        "uid_pssm_rate": float(uid_audit["has_pssm"].mean()) if len(uid_audit) else 0.0,
        "uid_dssp_rate": float(uid_audit["has_dssp"].mean()) if len(uid_audit) else 0.0,
        "uid_seq_pssm_dssp_rate": float((uid_audit["in_seq"] & uid_audit["has_pssm"] & uid_audit["has_dssp"]).mean()) if len(uid_audit) else 0.0,
        "missing_pssm_uid_count": int((~uid_audit["has_pssm"]).sum()) if len(uid_audit) else 0,
        "missing_dssp_uid_count": int((~uid_audit["has_dssp"]).sum()) if len(uid_audit) else 0,
    }

    summary = {}
    summary.update({"root": str(root), "out_root": str(out_root)})
    summary.update(dir_summary)
    summary.update(file_summary)
    summary.update(fail_summary)
    summary.update(seq_summary)
    summary.update({f"pssm_{k}": v for k, v in pssm_summary.items() if k != "path"})
    summary["pssm_path"] = pssm_summary.get("path", "")
    summary.update({f"dssp_{k}": v for k, v in dssp_summary.items() if k != "path"})
    summary["dssp_path"] = dssp_summary.get("path", "")
    summary.update({f"shapes_{k}": v for k, v in shapes_summary.items() if k != "path"})
    summary["shapes_path"] = shapes_summary.get("path", "")
    summary.update({f"pairs_{k}": v for k, v in pairs_summary.items() if k != "path"})
    summary["pairs_path"] = pairs_summary.get("path", "")
    summary.update({f"contacts_{k}": v for k, v in contacts_summary.items() if k != "path"})
    summary["contacts_path"] = contacts_summary.get("path", "")
    summary.update(coverage_summary)

    copy_key_files(root, out_root, copy_required_dirs=not bool(args.no_copy_data))
    repair_summary = {"enabled": False}
    if not bool(args.no_repair_dssp):
        repair_summary = repair_dssp_table(
            src_path=root / "fourpack" / "tuna.all.dssp.tsv",
            fasta_path=root / "fourpack" / "tuna.seq.fasta",
            struct_dir=root / "struct",
            out_path=out_root / "prepared" / "fourpack" / "tuna.all.dssp.tsv",
            coords_dir=out_root / "prepared" / "coords",
            report_path=audit / "tuna_dssp_repair_by_uid.tsv",
        )
        summary.update({f"repair_dssp_{k}": v for k, v in repair_summary.items()})
        repaired_by_uid, repaired_summary = summarize_dssp(out_root / "prepared" / "fourpack" / "tuna.all.dssp.tsv", seq_lens)
        if len(repaired_by_uid):
            repaired_by_uid.to_csv(audit / "tuna_dssp_repaired_quality_by_uid.tsv", sep="\t", index=False)
        summary.update({f"repaired_dssp_{k}": v for k, v in repaired_summary.items() if k != "path"})
        summary["repaired_dssp_path"] = repaired_summary.get("path", "")

    write_summary_tsv(summary, audit / "tuna_audit_summary.tsv")
    with (audit / "tuna_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_root": str(root),
            "out_root": str(out_root),
            "prepared_root": str(out_root / "prepared"),
            "audit_dir": str(audit),
            "copied_required_data": not bool(args.no_copy_data),
            "copied_dirs": list(EXPECTED_DIRS) if not bool(args.no_copy_data) else [],
            "dssp_repair": repair_summary,
        }, f, ensure_ascii=False, indent=2)

    print("[TUnA] audit finished", flush=True)
    for key in (
        "n_root_pairs",
        "n_pair_shapes",
        "n_expected_uid",
        "seq_n_uid",
        "uid_seq_rate",
        "uid_pssm_rate",
        "uid_dssp_rate",
        "uid_seq_pssm_dssp_rate",
        "pssm_mean_abs_mean",
        "pssm_mean_nonzero_rate",
        "pssm_near_zero_uid_rate",
        "dssp_mean_ss_valid_rate",
        "dssp_mean_ss_noncoil_rate",
        "dssp_rsa_all_zero_uid_rate",
        "dssp_asa_all_zero_uid_rate",
        "repair_dssp_n_uid_repaired_rsa",
        "repair_dssp_n_uid_with_coords",
        "repair_dssp_n_uid_coord_len_match",
        "repair_dssp_rsa_all_zero_after",
        "repair_dssp_rsa_mean_after",
        "repaired_dssp_rsa_all_zero_uid_rate",
        "repaired_dssp_asa_all_zero_uid_rate",
        "contacts_n_rows",
    ):
        if key in summary:
            print(f"[TUnA] {key}={summary[key]}", flush=True)
    print(f"[TUnA] out_root={out_root}", flush=True)
    print(f"[TUnA] audit_dir={audit}", flush=True)


if __name__ == "__main__":
    main()
