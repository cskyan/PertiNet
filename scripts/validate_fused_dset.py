"""Validate the fused Dset source configured for the manuscript experiment."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import DSET_ROOT
from model.dset import load_fused_dset


EXPECTED_SPLIT_COUNTS = {"train": 302, "validation": 50, "test": 70}


def validate(root):
    bundle = load_fused_dset(root)
    n_records = bundle["n_records"]
    structural_length_adjustments = []
    core_length_mismatches = []
    for index in range(n_records):
        lengths = {
            "sequence": len(bundle["sequence"][index]),
            "pssm": len(bundle["pssm"][index]),
            "dssp": len(bundle["dssp"][index]),
            "label": len(bundle["label"][index]),
        }
        if len({lengths["sequence"], lengths["pssm"], lengths["label"]}) != 1:
            core_length_mismatches.append({"index": index, **lengths})
        if lengths["dssp"] != lengths["sequence"]:
            structural_length_adjustments.append({"index": index, **lengths})

    split_sets = {
        split: set(bundle[split]) for split in ("train", "validation", "test")
    }
    overlap = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"fused split indices overlap: {overlap}")
    if core_length_mismatches:
        raise ValueError(
            "fused sequence/PSSM/label lengths do not align: "
            f"{core_length_mismatches[:5]}"
        )

    assigned = set().union(*split_sets.values())
    split_counts = {key: len(value) for key, value in split_sets.items()}
    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "expected the DeepPPISP-referenced 302/50/70 protein split, "
            f"observed {split_counts}"
        )
    if assigned != set(range(n_records)):
        raise ValueError("the split lists do not cover every fused record exactly once")
    return {
        "data_root": str(Path(root)),
        "records": n_records,
        "split_counts": split_counts,
        "assigned_records": len(assigned),
        "unassigned_records": n_records - len(assigned),
        "split_overlap": overlap,
        "sequence_pssm_label_mismatches": 0,
        "dssp_records_aligned_to_common_length": len(structural_length_adjustments),
        "maximum_dssp_length_difference": max(
            (abs(item["dssp"] - item["sequence"]) for item in structural_length_adjustments),
            default=0,
        ),
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DSET_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root), indent=2))


if __name__ == "__main__":
    main()
