import os
import argparse
import pandas as pd


def build_topk(model_dir: str,
               centers: list,
               ks=(50, 30, 10),
               csv_name="test_pairs_with_disturb_scores.csv") -> None:
    """
    Read the final prediction CSV and produce:
      1) top{K}_ILF3_PTBP1_disturb.csv  (Protein_A, Protein_B, disturb_score)
      2) top_ILF3_PTBP1_nodes_{K}.txt   (unique node list, one ID per line)

    Parameters
    ----------
    model_dir : str
        Folder that contains the prediction CSV.
    centers : list
        Center UniProt IDs; any edge touching one of these is kept.
        Defaults target ILF3 (Q12906) and PTBP1 (P26599).
    ks : tuple
        K values to export (descending by disturb_score).
    csv_name : str
        File name of the prediction CSV produced by predict.py.
    """
    in_csv = os.path.join(model_dir, csv_name)
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"File not found: {in_csv}")

    df = pd.read_csv(in_csv)

    required = {"Protein_A", "Protein_B", "disturb_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    mask = df["Protein_A"].isin(centers) | df["Protein_B"].isin(centers)
    sub = df.loc[mask].copy()
    sub.sort_values("disturb_score", ascending=False, inplace=True)

    for k in ks:
        topk = sub.head(k).copy()

        out_csv = os.path.join(model_dir, f"top{k}_ILF3_PTBP1_disturb.csv")
        topk[["Protein_A", "Protein_B", "disturb_score"]].to_csv(out_csv, index=False)

        nodes = pd.unique(pd.concat([topk["Protein_A"], topk["Protein_B"]], ignore_index=True))
        out_nodes = os.path.join(model_dir, f"top_ILF3_PTBP1_nodes_{k}.txt")
        with open(out_nodes, "w") as f:
            for n in nodes:
                f.write(str(n).strip() + "\n")

        print(f"[OK] K={k}: {out_csv} | {out_nodes}")

    print(f"Done. Source: {in_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Top-K ILF3/PTBP1 files for picture.py from a prediction CSV."
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Directory containing the prediction CSV."
    )
    parser.add_argument(
        "--centers", nargs="*", default=["Q12906", "P26599"],
        help="Center UniProt IDs (default: ILF3=Q12906, PTBP1=P26599)."
    )
    parser.add_argument(
        "--k", nargs="*", type=int, default=[50, 30, 10],
        help="List of K values, e.g., --k 100 50 20 (default: 50 30 10)."
    )
    parser.add_argument(
        "--csv_name", default="test_pairs_with_disturb_scores.csv",
        help="Prediction CSV file name (default: test_pairs_with_disturb_scores.csv)."
    )
    args = parser.parse_args()
    build_topk(args.model_dir, args.centers, tuple(args.k), args.csv_name)


if __name__ == "__main__":
    main()
