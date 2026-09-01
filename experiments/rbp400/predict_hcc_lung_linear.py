#!/usr/bin/env python3
"""RBP400 HCC/lung candidate ranking with a case-excluded sequence ensemble.

The companion shell workflow first refits five development models on the
case-excluded RBP400-Expanded pairs and then calls this script to score the two
predefined RBP400 candidate networks. RBP400-Expanded supplies controlled
pair-level labels, whereas RBP400 defines the downstream candidate and
interpretation space. HCC and lung are context-specific candidate universes,
not separately trained disease models or public benchmarks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .train_esm_case_model import load_embedding
except ImportError:
    from train_esm_case_model import load_embedding


ANCHORS = {"HCC": {"Q12906", "P26599"}, "lung": {"O00425", "P22626"}}
EXPECTED_COUNTS = {"HCC": 16471, "lung": 27261, "shared": 5151, "models": 5}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rbp400-root", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--smoke-pairs", type=int, default=0)
    parser.add_argument(
        "--allow-nonstandard-counts",
        action="store_true",
        help="Allow candidate/model counts that differ from the locked RBP400 case-study manifest.",
    )
    return parser.parse_args()


def canonical(a, b):
    return "__".join(sorted((str(a).strip(), str(b).strip())))


def load_context(root, filename, context):
    frame = pd.read_csv(root / "pairs_strict" / filename, sep="\t", dtype=str)
    required = {"protein_A", "protein_B"}
    if not required.issubset(frame.columns):
        raise ValueError("%s lacks protein_A/protein_B" % filename)
    frame["pair_id"] = [canonical(a, b) for a, b in zip(frame.protein_A, frame.protein_B)]
    frame["context"] = context
    if frame.pair_id.duplicated().any():
        raise RuntimeError("Duplicate pairs in %s" % filename)
    return frame


def load_models(root):
    models = []
    for path in sorted(root.glob("seed_*/linear_model.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        required = {"protein_scaler", "pca", "pair_scaler", "classifier", "pooling", "threshold"}
        missing = required.difference(payload)
        if missing:
            raise KeyError("%s missing keys: %s" % (path, sorted(missing)))
        seed = path.parent.name.replace("seed_", "")
        models.append((seed, payload, path))
    if not models:
        raise FileNotFoundError("No seed_*/linear_model.pkl below %s" % root)
    poolings = {payload["pooling"] for _, payload, _ in models}
    if len(poolings) != 1:
        raise RuntimeError("All ensemble members must use one pooling strategy")
    return models, next(iter(poolings))


def load_development_metrics(model_root):
    path = model_root / "metrics_summary.json"
    if not path.exists():
        raise FileNotFoundError("Missing development metrics: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_path(path, label):
    if not path.exists():
        raise FileNotFoundError("Missing %s: %s" % (label, path))


def transform_proteins(raw_matrix, proteins, payload):
    scaled = payload["protein_scaler"].transform(raw_matrix)
    low = payload["pca"].transform(scaled).astype(np.float32)
    return {protein: vector for protein, vector in zip(proteins, low)}


def pair_features(frame, vectors):
    a = np.stack([vectors[p] for p in frame.protein_A])
    b = np.stack([vectors[p] for p in frame.protein_B])
    cosine = (a * b).sum(1, keepdims=True) / (
        np.linalg.norm(a, axis=1, keepdims=True)
        * np.linalg.norm(b, axis=1, keepdims=True)
        + 1e-8
    )
    return np.concatenate([np.abs(a - b), a * b, 0.5 * (a + b), cosine], axis=1).astype(np.float32)


def score_one_model(frame, vectors, payload, batch_size):
    probabilities = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        features = pair_features(batch, vectors)
        features = payload["pair_scaler"].transform(features)
        probabilities.append(payload["classifier"].predict_proba(features)[:, 1])
    return np.concatenate(probabilities)


def score_context(frame, proteins, raw_matrix, models, batch_size):
    per_seed = []
    thresholds = []
    seed_names = []
    for seed, payload, _ in models:
        vectors = transform_proteins(raw_matrix, proteins, payload)
        per_seed.append(score_one_model(frame, vectors, payload, batch_size))
        thresholds.append(float(payload["threshold"]))
        seed_names.append(seed)
    matrix = np.stack(per_seed, axis=1)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite prediction detected")
    ddof = 1 if matrix.shape[1] > 1 else 0
    result = frame.copy()
    result["interaction_probability_mean"] = matrix.mean(axis=1)
    result["interaction_probability_sd"] = matrix.std(axis=1, ddof=ddof)
    result["ensemble_decision_threshold"] = float(np.mean(thresholds))
    result["confidence_margin"] = np.abs(
        result.interaction_probability_mean - result.ensemble_decision_threshold
    )
    result["rank_probability"] = result.interaction_probability_mean.rank(
        method="min", ascending=False
    ).astype(int)
    denominator = max(len(result) - 1, 1)
    result["rank_percentile"] = 1.0 - (result.rank_probability - 1) / denominator
    for column, seed in enumerate(seed_names):
        result["probability_seed_%s" % seed] = matrix[:, column]
    return result.sort_values("rank_probability").reset_index(drop=True), matrix


def mean_seed_spearman(matrix):
    if matrix.shape[1] < 2:
        return None
    corr = pd.DataFrame(matrix).corr(method="spearman").to_numpy()
    values = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(values))


def write_context(frame, output_root, context):
    prefix = context.lower()
    frame.to_csv(output_root / (prefix + "_all_scored.tsv"), sep="\t", index=False)
    anchors = ANCHORS[context]
    anchor_frame = frame[
        frame.protein_A.isin(anchors) | frame.protein_B.isin(anchors)
    ]
    anchor_frame.to_csv(output_root / (prefix + "_anchor_scored.tsv"), sep="\t", index=False)
    for k in (10, 30, 50, 100):
        frame.head(k).to_csv(output_root / (prefix + "_top%d.tsv" % k), sep="\t", index=False)


def validate_locked_counts(hcc, lung, models, allow_nonstandard_counts):
    observed = {"HCC": len(hcc), "lung": len(lung), "models": len(models)}
    mismatches = {
        key: {"expected": EXPECTED_COUNTS[key], "observed": value}
        for key, value in observed.items()
        if value != EXPECTED_COUNTS[key]
    }
    if mismatches and not allow_nonstandard_counts:
        raise RuntimeError("Locked RBP400 manifest mismatch: %s" % mismatches)
    return observed


def main():
    args = parse_args()
    started_at = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    require_path(args.rbp400_root / "pairs_strict" / "hcc_all_pairs.tsv", "HCC candidate pairs")
    require_path(args.rbp400_root / "pairs_strict" / "lung_all_pairs.tsv", "lung candidate pairs")
    require_path(args.esm_cache, "ESM cache")
    require_path(args.model_root, "frozen development model directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    hcc = load_context(args.rbp400_root, "hcc_all_pairs.tsv", "HCC")
    lung = load_context(args.rbp400_root, "lung_all_pairs.tsv", "lung")
    if args.smoke_pairs > 0:
        hcc = hcc.head(args.smoke_pairs).copy()
        lung = lung.head(args.smoke_pairs).copy()

    models, pooling = load_models(args.model_root)
    if args.smoke_pairs == 0:
        validate_locked_counts(hcc, lung, models, args.allow_nonstandard_counts)
    development_metrics = load_development_metrics(args.model_root)
    proteins = sorted(set(hcc.protein_A) | set(hcc.protein_B) | set(lung.protein_A) | set(lung.protein_B))
    pooled = {protein: load_embedding(args.esm_cache, protein, pooling).numpy() for protein in proteins}
    raw_matrix = np.stack([pooled[protein] for protein in proteins]).astype(np.float32)

    hcc_scored, hcc_matrix = score_context(hcc, proteins, raw_matrix, models, args.batch_size)
    lung_scored, lung_matrix = score_context(lung, proteins, raw_matrix, models, args.batch_size)
    shared = set(hcc_scored.pair_id) & set(lung_scored.pair_id)
    hcc_scored["shared_pair"] = hcc_scored.pair_id.isin(shared).astype(int)
    lung_scored["shared_pair"] = lung_scored.pair_id.isin(shared).astype(int)
    write_context(hcc_scored, args.output_root, "HCC")
    write_context(lung_scored, args.output_root, "lung")

    shared_table = hcc_scored[hcc_scored.shared_pair == 1][
        ["pair_id", "interaction_probability_mean", "rank_probability", "rank_percentile"]
    ].merge(
        lung_scored[lung_scored.shared_pair == 1][
            ["pair_id", "interaction_probability_mean", "rank_probability", "rank_percentile"]
        ],
        on="pair_id",
        suffixes=("_hcc", "_lung"),
    )
    shared_table["probability_difference_check"] = np.abs(
        shared_table.interaction_probability_mean_hcc
        - shared_table.interaction_probability_mean_lung
    )
    max_shared_difference = float(shared_table.probability_difference_check.max()) if len(shared_table) else 0.0
    if args.smoke_pairs == 0 and len(shared_table) != EXPECTED_COUNTS["shared"] and not args.allow_nonstandard_counts:
        raise RuntimeError(
            "Locked shared-pair count mismatch: expected %d, observed %d"
            % (EXPECTED_COUNTS["shared"], len(shared_table))
        )
    if max_shared_difference > 1e-6:
        raise RuntimeError("Shared-pair score inconsistency: max difference %.8g" % max_shared_difference)
    shared_table.to_csv(args.output_root / "shared_hcc_lung.tsv", sep="\t", index=False)

    hcc_spearman = mean_seed_spearman(hcc_matrix)
    lung_spearman = mean_seed_spearman(lung_matrix)
    case_summary = pd.DataFrame(
        [
            {
                "case_study": "RBP400",
                "context": "HCC",
                "candidate_pairs": len(hcc_scored),
                "unique_proteins": len(set(hcc_scored.protein_A) | set(hcc_scored.protein_B)),
                "shared_pairs": len(shared_table),
                "score_mean": float(hcc_scored.interaction_probability_mean.mean()),
                "score_median": float(hcc_scored.interaction_probability_mean.median()),
                "top10_score_mean": float(hcc_scored.head(10).interaction_probability_mean.mean()),
                "mean_seed_spearman": hcc_spearman,
            },
            {
                "case_study": "RBP400",
                "context": "lung",
                "candidate_pairs": len(lung_scored),
                "unique_proteins": len(set(lung_scored.protein_A) | set(lung_scored.protein_B)),
                "shared_pairs": len(shared_table),
                "score_mean": float(lung_scored.interaction_probability_mean.mean()),
                "score_median": float(lung_scored.interaction_probability_mean.median()),
                "top10_score_mean": float(lung_scored.head(10).interaction_probability_mean.mean()),
                "mean_seed_spearman": lung_spearman,
            },
        ]
    )
    case_summary.to_csv(args.output_root / "rbp400_case_summary.tsv", sep="\t", index=False)

    summary = {
        "status": "completed",
        "case_study_name": "RBP400",
        "started_at_utc": started_at,
        "completed_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "model_family": "RBP400-targeted PCA-logistic ESM ensemble",
        "resource_roles": {
            "model_development": "RBP400-Expanded (case-excluded pair-level resource)",
            "candidate_prioritization": "RBP400 (downstream candidate-protein resource)",
            "public_benchmark_claim": False,
        },
        "models": len(models),
        "pooling": pooling,
        "hcc_scored": len(hcc_scored),
        "lung_scored": len(lung_scored),
        "shared_scored": len(shared_table),
        "unique_case_proteins": len(proteins),
        "case_pairs_used_for_training": 0,
        "hcc_mean_seed_spearman": hcc_spearman,
        "lung_mean_seed_spearman": lung_spearman,
        "max_shared_probability_difference": max_shared_difference,
        "development_metrics": development_metrics,
        "locked_manifest": EXPECTED_COUNTS,
        "score_interpretation": "ensemble interaction-probability score",
        "confidence_interpretation": "distance from the validation-selected decision threshold",
        "context_interpretation": (
            "HCC and lung define candidate universes; they are not separately trained disease models. "
            "Shared pairs therefore have the same base probability but may have different within-network ranks."
        ),
    }
    (args.output_root / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
