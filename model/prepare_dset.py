# -*- coding: utf-8 -*-
"""Materialize GraphRBF-style Dset pickle files for PertiNet evaluation.

The training code reads a simple prepared layout:

    labels/<pid>.npy
    pssm/<pid>.npy
    dssp/<pid>.npy
    seq/<pid>.fasta
    train.txt / val.txt / test.txt

This script keeps the original Dest split semantics where possible: fused
training/validation lists for training, and dset186+dset72+dset164 as the
benchmark test split.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


DEFAULT_RAW_ROOT = "data/Dset"
DEFAULT_OUT_ROOT = "data/Dset_prepared"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_id(value) -> str:
    text = str(value).strip()
    text = text.split()[0] if text else ""
    text = re.sub(r"\.(npy|npz|pkl|fa|fasta|txt)$", "", text, flags=re.I)
    return text


def read_fasta_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    cur = ""
    parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    out[cur] = "".join(parts)
                cur = clean_id(line[1:])
                parts = []
            else:
                parts.append(line)
    if cur:
        out[cur] = "".join(parts)
    return out


def raw_fasta_map(raw_root: Path) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for name in ("all_seqs.fasta", "filtered_seqs.fasta", "dset72_sequences.fasta"):
        for pid, seq in read_fasta_map(raw_root / name).items():
            merged.setdefault(pid, seq)
    return merged


def looks_like_numeric_index_ids(ids: Sequence[str]) -> bool:
    if not ids:
        return False
    head = list(ids[: min(len(ids), 32)])
    n_num = sum(1 for pid in head if re.fullmatch(r"\d+", str(pid)))
    return n_num >= max(4, len(head) // 2)


ID_KEYS = (
    "id", "ids", "pid", "pids", "name", "names", "protein", "proteins",
    "protein_id", "protein_ids", "uid", "uids", "pdb", "pdbs", "pdbid",
    "pdbids", "pdb_id", "pdb_ids", "uniprot", "uniprot_id", "uniprot_ids",
    "seq_id", "seq_ids", "sequence_id", "sequence_ids", "all_seq_name",
    "seq_name", "seq_names", "protein_name", "protein_names", "list",
)
VALUE_KEYS = {
    "label": ("label", "labels", "y", "ys", "target", "targets", "data"),
    "pssm": ("pssm", "pssm_data", "feature", "features", "data"),
    "dssp": ("dssp", "dssp_data", "ss", "rsa", "feature", "features", "data"),
    "sequence": ("seq", "seqs", "sequence", "sequences", "fasta", "data"),
}


def is_sequence_like(value) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        len(value)
        return True
    except Exception:
        return False


def dataframe_to_mapping(obj):
    if hasattr(obj, "columns") and hasattr(obj, "__getitem__"):
        try:
            return {str(col): list(obj[col]) for col in list(obj.columns)}
        except Exception:
            return obj
    return obj


def safe_len(value) -> int:
    if isinstance(value, (str, bytes)):
        return -1
    try:
        return len(value)
    except Exception:
        return -1


def first_item(value):
    try:
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 0:
                return None
            return value.reshape(-1)[0]
        return list(value)[0]
    except Exception:
        return None


def scalar_string_score(value) -> float:
    item = first_item(value)
    if item is None:
        return 0.0
    if isinstance(item, bytes):
        return 1.0
    if isinstance(item, str):
        return 1.0
    arr = np.asarray(item)
    return 0.0 if arr.ndim > 0 else 0.3


def feature_score(value, kind: str) -> float:
    item = first_item(value)
    if item is None:
        return 0.0
    if kind == "sequence":
        if isinstance(item, (str, bytes)):
            return 1.0
        arr = np.asarray(item)
        return 0.7 if arr.ndim >= 1 else 0.0
    arr = np.asarray(item)
    if arr.dtype.kind in {"U", "S", "O"}:
        try:
            arr.astype(np.float32)
        except Exception:
            return 0.0
    return 1.0 if arr.size > 0 else 0.0


def as_id_list(obj) -> List[str]:
    obj = dataframe_to_mapping(obj)
    if isinstance(obj, Mapping):
        for key in ID_KEYS:
            if key in obj:
                vals = obj[key]
                break
        else:
            vals = obj.keys()
    else:
        vals = obj
    out: List[str] = []
    seen = set()
    for item in vals:
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        pid = clean_id(item)
        if pid and pid not in seen:
            out.append(pid)
            seen.add(pid)
    return out


def aligned_mapping_from_columns(obj: Mapping, kind: str) -> Dict[str, object]:
    obj = dataframe_to_mapping(obj)
    id_values = None
    value_values = None
    lower = {str(k).lower(): k for k in obj.keys()}
    for key in ID_KEYS:
        if key in lower and is_sequence_like(obj[lower[key]]):
            id_values = obj[lower[key]]
            break
    for key in VALUE_KEYS.get(kind, ("data",)):
        if key in lower and is_sequence_like(obj[lower[key]]):
            value_values = obj[lower[key]]
            break
    if id_values is None or value_values is None:
        columns = [(k, obj[k], safe_len(obj[k])) for k in obj.keys()]
        columns = [(k, v, n) for k, v, n in columns if n > 0]
        best = None
        for id_key, id_col, n in columns:
            id_score = scalar_string_score(id_col)
            if id_score <= 0:
                continue
            for val_key, val_col, n2 in columns:
                if id_key == val_key or n2 != n:
                    continue
                val_score = feature_score(val_col, kind)
                if val_score <= 0:
                    continue
                name_bonus = 0.0
                lk = str(val_key).lower()
                if any(token in lk for token in VALUE_KEYS.get(kind, ())):
                    name_bonus += 1.0
                if any(token in str(id_key).lower() for token in ID_KEYS):
                    name_bonus += 1.0
                score = id_score + val_score + name_bonus
                if best is None or score > best[0]:
                    best = (score, id_col, val_col, id_key, val_key)
        if best is None:
            return {}
        _, id_values, value_values, id_key, val_key = best
        print(f"[dest-prepare] inferred columns kind={kind} id={id_key} value={val_key} n={safe_len(id_values)}", flush=True)
    ids = list(id_values)
    values = list(value_values)
    if len(ids) != len(values):
        return {}
    return {clean_id(pid): val for pid, val in zip(ids, values) if clean_id(pid)}


def normalize_feature_map(obj, kind: str, ids: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Convert common Dest pickle shapes into pid -> feature."""
    obj = dataframe_to_mapping(obj)
    if isinstance(obj, Mapping):
        col = aligned_mapping_from_columns(obj, kind)
        if col:
            return col
        for key in VALUE_KEYS.get(kind, ("data",)):
            if key in obj and isinstance(obj[key], Mapping):
                return normalize_feature_map(obj[key], kind)
        if len(obj) <= 8:
            for value in obj.values():
                if isinstance(value, Mapping):
                    nested = normalize_feature_map(value, kind)
                    if nested:
                        return nested
        out = {}
        for key, value in obj.items():
            pid = clean_id(key)
            if pid:
                out[pid] = value
        return out
    if isinstance(obj, (list, tuple)):
        if ids is not None and len(ids) == len(obj):
            return {clean_id(pid): val for pid, val in zip(ids, obj) if clean_id(pid)}
        out = {}
        if len(obj) == 2 and is_sequence_like(obj[0]) and is_sequence_like(obj[1]):
            ids = list(obj[0])
            values = list(obj[1])
            if len(ids) == len(values):
                return {clean_id(pid): val for pid, val in zip(ids, values) if clean_id(pid)}
        for item in obj:
            if isinstance(item, Mapping):
                key = None
                lower = {str(k).lower(): k for k in item.keys()}
                for cand in ID_KEYS:
                    if cand in lower:
                        key = item[lower[cand]]
                        break
                value = None
                for cand in VALUE_KEYS.get(kind, ("data",)):
                    if cand in lower:
                        value = item[lower[cand]]
                        break
                if key is not None and value is not None:
                    out[clean_id(key)] = value
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out[clean_id(item[0])] = item[1]
        return {k: v for k, v in out.items() if k}
    return {}


def to_numeric_array(value, kind: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("label", "labels", "y", "pssm", "dssp", "feature", "data", kind):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (list, tuple)) and value:
        candidates = []
        for item in value:
            try:
                arr = np.asarray(item)
            except Exception:
                continue
            if arr.dtype.kind in {"U", "S", "O"}:
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    continue
            if arr.size > 0:
                candidates.append(arr.astype(np.float32))
        if candidates:
            candidates.sort(key=lambda x: (x.ndim, x.size), reverse=True)
            value = candidates[0]
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        try:
            arr = arr.astype(np.float32)
        except Exception:
            return None
    arr = arr.astype(np.float32)
    if kind == "labels":
        return normalize_label_array(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def normalize_label_array(arr: np.ndarray, target_len: int = 0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(-1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if target_len > 0:
        if arr.shape[0] == target_len and arr.shape[1] <= 4:
            return arr[:, -1].reshape(-1)
        if arr.shape[1] == target_len and arr.shape[0] <= 4:
            return arr[-1, :].reshape(-1)
    if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
        return arr[-1, :].reshape(-1)
    if arr.shape[1] <= 4 and arr.shape[0] > arr.shape[1]:
        return arr[:, -1].reshape(-1)
    return arr.reshape(-1)


def orient_residue_feature(arr: Optional[np.ndarray], target_len: int = 0) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return arr.reshape(arr.shape[0], -1)
    if target_len > 0:
        if arr.shape[0] == target_len:
            return arr
        if arr.shape[1] == target_len:
            return arr.T
    if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
        return arr.T
    return arr


def to_sequence(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("seq", "sequence", "fasta", "data"):
            if key in value:
                return to_sequence(value[key])
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        lines = [x.strip() for x in value.splitlines() if x.strip() and not x.startswith(">")]
        return "".join(lines)
    if isinstance(value, (list, tuple)) and value:
        seq_candidates = []
        for item in value:
            seq = to_sequence(item)
            if seq:
                seq_candidates.append(seq)
        if seq_candidates:
            seq_candidates.sort(key=len, reverse=True)
            return seq_candidates[0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item()).strip()
    chars = []
    for item in arr.reshape(-1).tolist():
        s = str(item).strip()
        if len(s) == 1 and s.isalpha():
            chars.append(s)
    return "".join(chars)


def prefix_ordered_ids(raw_root: Path, prefix: str, n_expected: int = 0) -> List[str]:
    fasta = raw_fasta_map(raw_root)
    if fasta and (not n_expected or len(fasta) == n_expected):
        ids = list(fasta.keys())
        print(f"[dest-prepare] id-order prefix={prefix} source=fasta n={len(ids)}", flush=True)
        return ids
    preferred_all = []
    if prefix == "fused":
        preferred_all.extend(["all_set_list.pkl", "all_dset_list.pkl"])
    preferred_all.extend([f"{prefix}_all_list.pkl", f"{prefix}_list.pkl"])
    for name in preferred_all:
        path = raw_root / name
        if path.exists():
            ids = as_id_list(load_pickle(path))
            if not n_expected or len(ids) == n_expected:
                if looks_like_numeric_index_ids(ids):
                    print(f"[dest-prepare][warn] ignore numeric index id-order source={name} n={len(ids)}", flush=True)
                else:
                    print(f"[dest-prepare] id-order prefix={prefix} source={name} n={len(ids)}", flush=True)
                    return ids
    names = [
        f"{prefix}_training_list.pkl",
        f"{prefix}_validing_list.pkl",
        f"{prefix}_validation_list.pkl",
        f"{prefix}_test_list.pkl",
    ]
    out: List[str] = []
    seen = set()
    for name in names:
        path = raw_root / name
        if not path.exists():
            continue
        for pid in as_id_list(load_pickle(path)):
            if pid and pid not in seen:
                out.append(pid)
                seen.add(pid)
    if n_expected and len(out) != n_expected:
        alt_names = ["all_set_list.pkl", "all_dset_list.pkl", f"{prefix}_list.pkl"]
        for name in alt_names:
            path = raw_root / name
            if path.exists():
                ids = as_id_list(load_pickle(path))
                if len(ids) == n_expected:
                    if looks_like_numeric_index_ids(ids):
                        continue
                    return ids
    return out


def load_existing_maps(raw_root: Path, prefixes: Sequence[str], suffix: str, kind: str) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for prefix in prefixes:
        path = raw_root / f"{prefix}_{suffix}.pkl"
        if not path.exists():
            continue
        obj = load_pickle(path)
        ordered_ids = prefix_ordered_ids(raw_root, prefix, safe_len(obj))
        sub = normalize_feature_map(obj, kind, ids=ordered_ids if len(ordered_ids) == safe_len(obj) else None)
        if len(sub) <= 2 and len(ordered_ids) == safe_len(obj):
            sub = {clean_id(pid): val for pid, val in zip(ordered_ids, obj) if clean_id(pid)}
        print(f"[dest-prepare] loaded {path.name} entries={len(sub)}", flush=True)
        merged.update(sub)
    return merged


def load_split(raw_root: Path, names: Sequence[str]) -> List[str]:
    for name in names:
        path = raw_root / name
        if path.exists():
            return as_id_list(load_pickle(path))
    return []


def describe_pickle(path: Path) -> str:
    if not path.exists():
        return f"{path.name}\tmissing"
    obj = load_pickle(path)
    typ = type(obj).__name__
    if isinstance(obj, Mapping):
        keys = list(obj.keys())
        preview = ",".join(str(k) for k in keys[:8])
        lens = []
        for k in keys[:8]:
            try:
                item = first_item(obj[k])
                item_shape = getattr(np.asarray(item), "shape", "")
                lens.append(f"{k}:{len(obj[k])}:first={type(item).__name__}:shape={item_shape}")
            except Exception:
                lens.append(f"{k}:NA")
        return f"{path.name}\t{typ}\tkeys={preview}\tlens={';'.join(lens)}"
    try:
        n = len(obj)
    except Exception:
        n = "NA"
    details = []
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(list(obj)[:5]):
            item_type = type(item).__name__
            item_len = safe_len(item)
            item_shape = getattr(np.asarray(item), "shape", "")
            if isinstance(item, (list, tuple)):
                parts = []
                for j, part in enumerate(list(item)[:6]):
                    part_arr = np.asarray(part)
                    preview = str(part)
                    if len(preview) > 40:
                        preview = preview[:37] + "..."
                    parts.append(f"{j}:{type(part).__name__}:shape={part_arr.shape}:val={preview}")
                details.append(f"item{idx}={item_type}:len={item_len}:shape={item_shape}:parts[{'; '.join(parts)}]")
            else:
                preview = str(item)
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                details.append(f"item{idx}={item_type}:len={item_len}:shape={item_shape}:val={preview}")
    return f"{path.name}\t{typ}\tlen={n}\t" + "\t".join(details)


def write_ids(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for pid in ids:
            f.write(f"{pid}\n")


def write_fasta(path: Path, pid: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f">{pid}\n")
        if seq:
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def unique_existing(ids: Iterable[str], available: set) -> List[str]:
    out = []
    seen = set()
    for pid in ids:
        if pid in available and pid not in seen:
            out.append(pid)
            seen.add(pid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Dset pickle files for PertiNet residue-level evaluation.")
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--test-prefixes", default="dset186,dset72,dset164")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    prefixes = ["fused", "low_homo", "dset186", "dset72", "dset164"]
    test_prefixes = [x.strip() for x in args.test_prefixes.split(",") if x.strip()]

    if args.inspect_only:
        for fasta_name in ("all_seqs.fasta", "filtered_seqs.fasta", "dset72_sequences.fasta"):
            fmap = read_fasta_map(raw_root / fasta_name)
            print(f"{fasta_name}\tfasta\tcount={len(fmap)}\tids={','.join(list(fmap.keys())[:8])}", flush=True)
        for prefix in prefixes:
            for suffix in ("label", "pssm_data", "dssp_data", "sequence_data", "training_list", "validing_list", "test_list"):
                print(describe_pickle(raw_root / f"{prefix}_{suffix}.pkl"), flush=True)
        return

    labels = load_existing_maps(raw_root, prefixes, "label", "label")
    pssm = load_existing_maps(raw_root, prefixes, "pssm_data", "pssm")
    dssp = load_existing_maps(raw_root, prefixes, "dssp_data", "dssp")
    seqs = load_existing_maps(raw_root, prefixes, "sequence_data", "sequence")
    fasta_seqs = raw_fasta_map(raw_root)

    if not labels:
        raise RuntimeError(f"No label pickle data found under {raw_root}")
    if len(labels) < 20:
        raise RuntimeError(
            f"Only parsed {len(labels)} labeled Dest proteins from {raw_root}; "
            "the pickle layout is not recognized yet. Run with --inspect-only and send the output."
        )
    if looks_like_numeric_index_ids(list(labels.keys())):
        raise RuntimeError(
            "Parsed Dest IDs look like numeric indices (0,1,2,...) rather than protein IDs. "
            "Run with --inspect-only and send the output; the list item layout needs explicit decoding."
        )

    for sub in ("labels", "pssm", "dssp", "seq"):
        subdir = out_root / sub
        subdir.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.npy", "*.npz", "*.fasta", "*.fa", "*.txt"):
            for old in subdir.glob(pattern):
                old.unlink()

    written = []
    skipped = []
    seq_len_match = 0
    seq_nonempty = 0
    pssm_len_match = 0
    dssp_len_match = 0
    length_examples = []
    for pid, label_value in sorted(labels.items()):
        seq = fasta_seqs.get(pid) or to_sequence(seqs.get(pid))
        y_raw = to_numeric_array(label_value, "labels")
        target_len = len(seq) if seq else 0
        y = normalize_label_array(y_raw, target_len=target_len) if y_raw is not None else None
        if y is None or y.size == 0:
            skipped.append((pid, "label"))
            continue
        np.save(out_root / "labels" / f"{pid}.npy", y.astype(np.float32))
        pssm_arr = orient_residue_feature(to_numeric_array(pssm.get(pid), "pssm"), target_len=int(y.shape[0]))
        if pssm_arr is not None:
            if int(pssm_arr.shape[0]) == int(y.shape[0]):
                pssm_len_match += 1
            np.save(out_root / "pssm" / f"{pid}.npy", pssm_arr.astype(np.float32))
        dssp_arr = orient_residue_feature(to_numeric_array(dssp.get(pid), "dssp"), target_len=int(y.shape[0]))
        if dssp_arr is not None:
            if int(dssp_arr.shape[0]) == int(y.shape[0]):
                dssp_len_match += 1
            np.save(out_root / "dssp" / f"{pid}.npy", dssp_arr.astype(np.float32))
        if len(length_examples) < 8:
            length_examples.append((
                pid,
                int(y.shape[0]),
                int(len(seq)) if seq else 0,
                int(pssm_arr.shape[0]) if pssm_arr is not None and pssm_arr.ndim >= 1 else 0,
                int(dssp_arr.shape[0]) if dssp_arr is not None and dssp_arr.ndim >= 1 else 0,
            ))
        if seq:
            seq_nonempty += 1
            if len(seq) == int(y.shape[0]):
                seq_len_match += 1
        write_fasta(out_root / "seq" / f"{pid}.fasta", pid, seq)
        written.append(pid)

    available = set(written)
    train_ids = unique_existing(load_split(raw_root, ["fused_training_list.pkl", "training_list.pkl"]), available)
    val_ids = unique_existing(load_split(raw_root, ["fused_validing_list.pkl", "validing_list.pkl", "validation_list.pkl"]), available)
    fused_test = unique_existing(load_split(raw_root, ["fused_test_list.pkl", "test_list.pkl"]), available)

    benchmark_ids: List[str] = []
    for prefix in test_prefixes:
        ids = unique_existing(load_split(raw_root, [f"{prefix}_test_list.pkl", f"{prefix}_validing_list.pkl"]), available)
        if not ids and (raw_root / f"{prefix}_test_list.pkl").exists():
            label_path = raw_root / f"{prefix}_label.pkl"
            if label_path.exists():
                obj = load_pickle(label_path)
                ordered_ids = prefix_ordered_ids(raw_root, prefix, safe_len(obj))
                ids = unique_existing(ordered_ids, available)
        write_ids(out_root / f"test_{prefix}.txt", ids)
        benchmark_ids.extend(ids)

    if not train_ids or not val_ids:
        all_ids = sorted(available)
        n = len(all_ids)
        train_ids = train_ids or all_ids[: int(0.8 * n)]
        val_ids = val_ids or all_ids[int(0.8 * n) : int(0.9 * n)]
    test_ids = fused_test or unique_existing(benchmark_ids, available)

    write_ids(out_root / "all_ids.txt", sorted(available))
    write_ids(out_root / "train.txt", train_ids)
    write_ids(out_root / "val.txt", val_ids)
    write_ids(out_root / "test.txt", test_ids)

    report = out_root / "dest_prepare_report.tsv"
    with open(report, "w", encoding="utf-8", newline="\n") as f:
        f.write("field\tvalue\n")
        f.write(f"raw_root\t{raw_root}\n")
        f.write(f"out_root\t{out_root}\n")
        f.write(f"written_proteins\t{len(written)}\n")
        f.write(f"skipped_proteins\t{len(skipped)}\n")
        f.write(f"seq_nonempty\t{seq_nonempty}\n")
        f.write(f"seq_label_len_match\t{seq_len_match}\n")
        f.write(f"pssm_label_len_match\t{pssm_len_match}\n")
        f.write(f"dssp_label_len_match\t{dssp_len_match}\n")
        f.write(f"train\t{len(train_ids)}\n")
        f.write(f"val\t{len(val_ids)}\n")
        f.write(f"test\t{len(test_ids)}\n")
        for prefix in test_prefixes:
            n = sum(1 for _ in open(out_root / f"test_{prefix}.txt", encoding="utf-8")) if (out_root / f"test_{prefix}.txt").exists() else 0
            f.write(f"test_{prefix}\t{n}\n")
        for idx, (pid, y_len, seq_len, pssm_len, dssp_len) in enumerate(length_examples, start=1):
            f.write(f"length_example_{idx}\t{pid}|label={y_len}|seq={seq_len}|pssm={pssm_len}|dssp={dssp_len}\n")

    print(f"[dest-prepare] wrote={len(written)} skipped={len(skipped)} out={out_root}", flush=True)
    print(f"[dest-prepare] split train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)
    print(f"[dest-prepare] report={report}", flush=True)


if __name__ == "__main__":
    main()
