"""Validate the immutable Bernett gold-standard protein-pair partitions."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "Intra1": {"role": "train", "positive": 81_596, "negative": 81_596},
    "Intra0": {"role": "validation", "positive": 29_630, "negative": 29_630},
    "Intra2": {"role": "test", "positive": 26_024, "negative": 26_024},
}


def read_pairs(path):
    pairs = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected two protein identifiers")
        pairs.append(tuple(fields))
    return pairs


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root):
    report = {"dataset": "Bernett gold-standard dataset", "splits": {}}
    protein_sets = {}
    for split, expected in EXPECTED.items():
        pos_path = root / f"{split}_pos_rr.txt"
        neg_path = root / f"{split}_neg_rr.txt"
        positive = read_pairs(pos_path)
        negative = read_pairs(neg_path)
        if len(positive) != expected["positive"] or len(negative) != expected["negative"]:
            raise AssertionError(
                f"{split}: observed pos/neg={len(positive)}/{len(negative)}, "
                f"expected {expected['positive']}/{expected['negative']}"
            )
        pair_keys = [tuple(sorted(pair)) for pair in positive + negative]
        if len(pair_keys) != len(set(pair_keys)):
            raise AssertionError(f"{split}: duplicate or label-conflicting unordered pairs detected")
        protein_sets[split] = {protein for pair in pair_keys for protein in pair}
        report["splits"][split] = {
            "role": expected["role"],
            "positive_pairs": len(positive),
            "negative_pairs": len(negative),
            "total_pairs": len(pair_keys),
            "unique_proteins": len(protein_sets[split]),
            "sha256": {"positive": sha256(pos_path), "negative": sha256(neg_path)},
        }
    overlaps = {}
    for left, right in (("Intra1", "Intra0"), ("Intra1", "Intra2"), ("Intra0", "Intra2")):
        count = len(protein_sets[left] & protein_sets[right])
        overlaps[f"{left}__{right}"] = count
        if count:
            raise AssertionError(f"protein overlap between {left} and {right}: {count}")
    report["protein_overlap"] = overlaps
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/bernett"))
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root), indent=2))


if __name__ == "__main__":
    main()
