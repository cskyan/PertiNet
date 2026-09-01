#!/usr/bin/env python3
"""Build case-safe matched RBP400-Expanded pairs and cold-protein splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expanded-root", type=Path, required=True)
    p.add_argument("--rbp400-root", type=Path, required=True)
    p.add_argument("--cluster-id30", type=Path, required=True)
    p.add_argument("--cluster-id20", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--locked-seed", type=int, default=260809,
                   help="Pre-registered seed for the new full-train/locked-blind split.")
    p.add_argument("--trials", type=int, default=4096)
    return p.parse_args()


def canonical(a, b):
    return "__".join(sorted((str(a).strip(), str(b).strip())))


def case_pair_ids(root: Path):
    result = set()
    for name in ("hcc_all_pairs.tsv", "lung_all_pairs.tsv"):
        df = pd.read_csv(root / "pairs_strict" / name, sep="\t", dtype=str)
        result.update(canonical(a, b) for a, b in zip(df.protein_A, df.protein_B))
    return result


def fasta_length(path: Path):
    length = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                length += len(line.strip())
    return length


def add_features(df: pd.DataFrame, expanded_root: Path):
    proteins = sorted(set(df.protein_A) | set(df.protein_B))
    lengths = {p: fasta_length(expanded_root / "seq" / f"{p}.fa") for p in proteins}
    degree = pd.concat([df.protein_A, df.protein_B]).value_counts().to_dict()
    out = df.copy()
    la = out.protein_A.map(lengths).to_numpy(float)
    lb = out.protein_B.map(lengths).to_numpy(float)
    da = out.protein_A.map(degree).to_numpy(float)
    db = out.protein_B.map(degree).to_numpy(float)
    out["length_short"] = np.minimum(la, lb)
    out["length_long"] = np.maximum(la, lb)
    out["degree_low"] = np.minimum(da, db)
    out["degree_high"] = np.maximum(da, db)
    return out


def scale(values):
    med = np.median(values, axis=0)
    q75, q25 = np.percentile(values, [75, 25], axis=0)
    width = q75 - q25
    width[width == 0] = 1
    return (values - med) / width


def match_pairs(df: pd.DataFrame, seed: int):
    fields = ["length_short", "length_long", "degree_low", "degree_high"]
    rng = np.random.default_rng(seed)
    negative = df[df.label.astype(int) == 0].copy()
    selected = []
    for scope, neg in negative.groupby("expansion_scope", sort=True):
        pos = df[(df.label.astype(int) == 1) & (df.expansion_scope == scope)].copy()
        if len(pos) < len(neg):
            raise RuntimeError(f"Not enough case-safe positives in {scope}: {len(pos)} < {len(neg)}")
        joined = pd.concat([neg, pos], ignore_index=True)
        z = scale(np.log1p(joined[fields].to_numpy(float)))
        zn, zp = z[: len(neg)], z[len(neg) :]
        available = np.ones(len(pos), dtype=bool)
        chosen = []
        for idx in rng.permutation(len(neg)):
            candidates = np.flatnonzero(available)
            distance = np.square(zp[candidates] - zn[idx]).sum(axis=1)
            best = candidates[np.isclose(distance, distance.min())]
            picked = int(rng.choice(best))
            chosen.append(picked)
            available[picked] = False
        selected.append(pos.iloc[chosen])
    positive = pd.concat(selected, ignore_index=True)
    result = pd.concat([positive, negative], ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_clusters(path: Path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    pcol = "protein_id" if "protein_id" in df else "protein"
    ccol = "cluster_id" if "cluster_id" in df else "cluster"
    return dict(zip(df[pcol], df[ccol]))


def stable(value: str, seed: int):
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()[:16], 16)


def split_cold(df: pd.DataFrame, clusters, seed: int, trials: int):
    units = sorted({clusters.get(p, p) for p in set(df.protein_A) | set(df.protein_B)})
    pair_units = [(clusters.get(a, a), clusters.get(b, b)) for a, b in zip(df.protein_A, df.protein_B)]
    labels = df.label.astype(int).to_numpy()
    ratios = np.asarray([0.7, 0.15, 0.15])
    root = np.sqrt(ratios); root /= root.sum()
    best = None
    for trial in range(trials):
        ordered = sorted(units, key=lambda x: stable(x, seed + trial * 104729))
        c1 = round(len(ordered) * root[0]); c2 = round(len(ordered) * (root[0] + root[1]))
        assignment = {u: i for i, part in enumerate((ordered[:c1], ordered[c1:c2], ordered[c2:])) for u in part}
        row_split = np.asarray([assignment[a] if assignment[a] == assignment[b] else -1 for a, b in pair_units])
        indices = [np.flatnonzero(row_split == i) for i in range(3)]
        counts = np.asarray([[int((labels[x] == y).sum()) for y in (0, 1)] for x in indices])
        if np.any(counts < 10):
            continue
        retained = sum(map(len, indices)) / len(df)
        label_deviation = 0.0
        for label in (0, 1):
            observed = counts[:, label] / counts[:, label].sum()
            label_deviation += np.abs(observed - ratios).sum()
        prevalence = counts[:, 1] / counts.sum(axis=1)
        score = label_deviation + 0.25 * np.ptp(prevalence) - 0.20 * retained
        candidate = (score, -retained, indices, row_split)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No class-complete homology-disjoint split found")
    return best[2], best[3]


def write_splits(df, cluster_path, out, seed, trials):
    indices, row_split = split_cold(df, load_clusters(cluster_path), seed, trials)
    out.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")
    audit = {"input_pairs": len(df), "excluded_cross_partition": int((row_split < 0).sum()), "splits": {}}
    protein_sets = {}
    for name, idx in zip(names, indices):
        part = df.iloc[idx].copy()
        part.to_csv(out / f"{name}.tsv", sep="\t", index=False)
        proteins = set(part.protein_A) | set(part.protein_B)
        protein_sets[name] = proteins
        audit["splits"][name] = {
            "pairs": len(part), "positive": int(part.label.astype(int).sum()),
            "negative": int((part.label.astype(int) == 0).sum()),
            "prevalence": float(part.label.astype(int).mean()), "proteins": len(proteins),
        }
    audit["protein_overlap"] = {
        "train_val": len(protein_sets["train"] & protein_sets["val"]),
        "train_test": len(protein_sets["train"] & protein_sets["test"]),
        "val_test": len(protein_sets["val"] & protein_sets["test"]),
    }
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if any(audit["protein_overlap"].values()):
        raise RuntimeError("Protein leakage detected")
    return audit


def write_locked_full_split(df, cluster_path, out, seed, trials):
    """Use every retained training pair while balancing validation and blind test.

    Protein clusters remain disjoint. Matching is performed only within each already
    assigned partition, so it cannot move proteins across partitions or leak cases.
    """
    indices, row_split = split_cold(df, load_clusters(cluster_path), seed, trials)
    out.mkdir(parents=True, exist_ok=True)
    raw = {name: df.iloc[idx].copy().reset_index(drop=True)
           for name, idx in zip(("train", "val", "test"), indices)}
    final = {
        "train": raw["train"],
        "val": match_pairs(raw["val"], seed + 1),
        "test": match_pairs(raw["test"], seed + 2),
    }
    protein_sets = {}
    audit = {
        "seed": seed,
        "input_pairs": len(df),
        "excluded_cross_partition": int((row_split < 0).sum()),
        "policy": "all case-safe train pairs; matched validation and locked blind test",
        "splits": {},
    }
    for name in ("train", "val", "test"):
        raw[name].to_csv(out / (name + "_full.tsv"), sep="\t", index=False)
        final[name].to_csv(out / (name + ".tsv"), sep="\t", index=False)
        proteins = set(final[name].protein_A) | set(final[name].protein_B)
        protein_sets[name] = proteins
        labels = final[name].label.astype(int)
        audit["splits"][name] = {
            "raw_pairs": len(raw[name]), "used_pairs": len(final[name]),
            "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
            "prevalence": float(labels.mean()), "proteins": len(proteins),
        }
    audit["protein_overlap"] = {
        "train_val": len(protein_sets["train"] & protein_sets["val"]),
        "train_test": len(protein_sets["train"] & protein_sets["test"]),
        "val_test": len(protein_sets["val"] & protein_sets["test"]),
    }
    if any(audit["protein_overlap"].values()):
        raise RuntimeError("Protein leakage detected in locked split")
    blind_bytes = (out / "test.tsv").read_bytes()
    audit["blind_test_sha256"] = hashlib.sha256(blind_bytes).hexdigest()
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    a = args_parser(); a.output_root.mkdir(parents=True, exist_ok=True)
    full_path = a.expanded_root / "pairs" / "expanded_true_pairs_full.tsv"
    full = pd.read_csv(full_path, sep="\t")
    full["pair_id"] = [canonical(x, y) for x, y in zip(full.protein_A, full.protein_B)]
    case_ids = case_pair_ids(a.rbp400_root)
    safe = full[~full.pair_id.isin(case_ids)].copy()
    safe = add_features(safe, a.expanded_root)
    matched = match_pairs(safe, a.seed)
    safe.to_csv(a.output_root / "case_safe_full.tsv", sep="\t", index=False)
    matched.to_csv(a.output_root / "case_safe_matched.tsv", sep="\t", index=False)
    audit = {
        "resource_roles": {
            "model_development": "RBP400-Expanded",
            "candidate_prioritization": "RBP400",
            "public_benchmark_claim": False,
        },
        "full_pairs": len(full), "case_universe_pairs": len(case_ids),
        "excluded_case_overlap": len(full) - len(safe), "case_safe_full": len(safe),
        "matched_pairs": len(matched), "matched_positive": int(matched.label.sum()),
        "matched_negative": int((matched.label == 0).sum()),
        "scope_by_label": matched.groupby(["expansion_scope", "label"]).size().rename("count").reset_index().to_dict("records"),
    }
    audit["id30"] = write_splits(matched, a.cluster_id30, a.output_root / "splits" / "id30", a.seed, a.trials)
    audit["id20"] = write_splits(matched, a.cluster_id20, a.output_root / "splits" / "id20", a.seed, a.trials)
    audit["id30_locked_full_train"] = write_locked_full_split(
        safe, a.cluster_id30, a.output_root / "splits" / "id30_locked", a.locked_seed, a.trials
    )
    (a.output_root / "preparation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
