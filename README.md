# PertiNet

PertiNet is a multimodal framework for protein-protein interaction prediction.
It combines sequence, structure, and Gene Ontology features through
interaction-aware token fusion and produces one score for each protein pair.

The repository contains the model and the scripts used for training,
prediction, data preparation, and RBP400 candidate ranking. Processed data,
fixed splits, embeddings, checkpoints, and result tables are stored on Zenodo:

<https://doi.org/10.5281/zenodo.21320101>

## Highlights

- Sequence, structure, and GO feature branches
- Interaction-aware modality fusion
- Fixed TUnA train, validation, and test levels
- RBP400 ablation and candidate-ranking support

## Repository layout

```text
PertiNet/
├─ model/
│  ├─ model.py
│  ├─ losses.py
│  ├─ config.py
│  ├─ train.py
│  ├─ predict.py
│  ├─ tuna_utils.py
│  ├─ prepare_tuna.py
│  ├─ prepare_dset.py
│  └─ rbp400_case.py
├─ scripts/
│  └─ compute_esm.py
├─ environment.yml
├─ LICENSE
└─ README.md
```

## Data and checkpoints

Download the release package from Zenodo and keep it outside the source
repository. The package contains:

```text
data/
├─ TUnA/
│  ├─ Intra1/          training
│  ├─ Intra0/          validation
│  └─ Intra2/          test
├─ RBP400/
├─ RBP400_Expanded/
└─ Dset_prepared/

checkpoints/
├─ pertinet_engineering.pt
├─ pertinet_tuna_no_go.pt
├─ pertinet_tuna_go.pt
└─ rbp400_case_ensemble/
```

The Git repository does not include full datasets, ESM caches, checkpoints, or
generated results. Use the files from the matching Zenodo release when
reproducing a manuscript result.

## Installation

```bash
conda env create -f environment.yml
conda activate pertinet
```

The environment uses Python 3.10, PyTorch 2.2, PyTorch Geometric 2.5, NumPy,
pandas, scikit-learn, SciPy, Biopython, Matplotlib, and seaborn.

## TUnA benchmark

PertiNet uses the original TUnA levels without re-splitting them:

- Intra1 for training
- Intra0 for validation and threshold selection
- Intra2 for final testing

Prepare or audit each level separately:

```bash
python model/prepare_tuna.py \
  --root /path/to/TUnA/Intra1 \
  --out-root /path/to/prepared/TUnA/Intra1
```

Train on the fixed levels:

```bash
python model/train.py \
  --stage tuna_pair_finetune \
  --resume /path/to/checkpoints/pertinet_engineering.pt \
  --tuna-train-fourpack-dir /path/to/data/TUnA/Intra1/fourpack \
  --tuna-val-fourpack-dir /path/to/data/TUnA/Intra0/fourpack \
  --tuna-test-fourpack-dir /path/to/data/TUnA/Intra2/fourpack \
  --out-dir outputs/tuna
```

If one fixed-level option is supplied, all three are required. The loader checks
protein overlap across the levels and does not generate a new split.

Check prediction options with:

```bash
python model/predict.py --help
```

The GO-free and GO-assisted models use the same pairs and fixed levels. The GO
branch is the only difference between those two settings.

## RBP400

RBP400 is used for matched ablation and candidate ranking. It is not the primary
public benchmark. The case-ranking model is fitted on pair-excluded
RBP400-Expanded data before it is applied to RBP400 candidate pairs.

```bash
python model/rbp400_case.py \
  --candidate-pairs /path/to/rbp400_candidate_pairs.tsv \
  --checkpoint /path/to/checkpoint.pt \
  --rbp400-root /path/to/data/RBP400 \
  --out-tsv outputs/rbp400_pair_scores.tsv
```

The reported score and distance from the validation threshold are model
outputs. They are used to rank candidates and are not measurements of
biological perturbation.

## Dset interface evaluation

The Dset_186_72_PDB164 workflow is a residue-level interface evaluation. It is
kept separate from the TUnA protein-pair benchmark.

```bash
python model/prepare_dset.py --help
```

## ESM embeddings

```bash
python scripts/compute_esm.py --help
```

Store downloaded ESM weights and generated embeddings outside Git.

## Reproducibility

- Keep the TUnA Intra1, Intra0, and Intra2 roles unchanged.
- Select thresholds and checkpoints with validation data only.
- Do not use RBP400 candidate pairs for training or model selection.
- Record the dataset version, checkpoint, seed, and command for each run.

## License

See [LICENSE](LICENSE).
