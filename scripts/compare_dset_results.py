"""Print metric differences between a completed Dset run and the manuscript."""

import argparse
import json
from pathlib import Path


TARGETS = {
    "PertiNet-S standard": {
        "acc": 0.763, "precision": 0.413, "recall": 0.633,
        "f1": 0.512, "auprc": 0.523, "mcc": 0.361,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        type=Path,
        nargs="?",
        default=Path("results/dset/full_dset_results.json"),
    )
    parser.add_argument("--target", choices=TARGETS, default="PertiNet-S standard")
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    observed = result["test"]
    print("metric\trun\tmanuscript\tdifference")
    for metric, target in TARGETS[args.target].items():
        value = float(observed[metric])
        print(f"{metric}\t{value:.4f}\t{target:.4f}\t{value-target:+.4f}")


if __name__ == "__main__":
    main()
