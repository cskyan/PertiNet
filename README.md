# PertiNet: interaction-aware multimodal PPI prediction

PertiNet models protein interactions with separate sequence, structure and Gene
Ontology (GO) representations. Modality tokens interact through self-attention
before sample-specific softmax weighting. The repository contains the model
implementation, executable smoke examples, the full residue-level Dset workflow
and the RBP400 case-study scripts described in the manuscript.

## Environment

Create the supplied Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate pertinet
```

All paths used by the bundled smoke and Dset workflows are repository-relative.
The Python source contains no machine-specific server paths. Commands for data
that are not distributed with the repository use `...` as a placeholder; users
must replace each `...` with the corresponding local data directory or file.

## Dataset terminology

The primary protein-pair benchmark is the **Bernett gold-standard dataset**.
**TUnA is a prediction method evaluated on that dataset, not a dataset name.**
The released Bernett pair files retain the original fixed roles:

| Split | Role | Positive | Negative | Total |
|---|---|---:|---:|---:|
| Intra1 | training | 81,596 | 81,596 | 163,192 |
| Intra0 | validation and threshold selection | 29,630 | 29,630 | 59,260 |
| Intra2 | final test | 26,024 | 26,024 | 52,048 |

The pair lists under `data/bernett/` can be checked with:

```bash
python scripts/validate_bernett.py
```

## Smoke tests

Smoke tests confirm that the released code and data interface execute. They are
not benchmark training runs and must not be used as reported performance.

### Model-framework smoke test

This test runs PertiNet pair prediction, all four fusion controls, the complete
training objective and the PertiNet-S residue track through forward and backward
passes:

```bash
python examples/smoke_test.py
```

### Three-record Dset example

`data/Dset_smoke_example/` contains three real records, one from Dset_72, one
from PDBset_164 and one from Dset_186. Each record includes sequence, PSSM, DSSP,
binary interface labels and real C-alpha coordinates. This small directory is
only an executable example of the expected per-protein data layout.

```bash
python examples/dset_smoke.py
```

### Automated release tests

```bash
python -m unittest discover -s tests -v
```

These tests cover the Bernett fixed partitions, Dset smoke data, the
DeepPPISP-referenced full-Dset split, the validation-selected operating boundary
and the locked RBP400 summary.

## Full Dset_186_72_PDB164 run

The full fused Dset source is stored in:

```text
data/Dset_186_72_PDB164/source/
```

It contains 422 protein-chain records, their fused sequence, PSSM, DSSP and
interface-label pickle files, and 422 C-alpha coordinate files. The split follows
the released DeepPPISP 352-development/70-test protein assignment. The original
70 test proteins are unchanged. Fifty proteins are deterministically held out
from the development pool with seed 2026 for validation, leaving:

| File | Role | Proteins |
|---|---|---:|
| `fused_training_list.pkl` | training | 302 |
| `fused_validing_list.pkl` | validation and threshold selection | 50 |
| `fused_test_list.pkl` | final held-out test | 70 |

The three sets are mutually disjoint and cover all 422 records.

### Step 1: validate the fused source

```bash
python scripts/validate_fused_dset.py
```

### Step 2: materialize per-protein files

```bash
python scripts/prepare_dset.py
```

This creates the generated directory
`data/Dset_186_72_PDB164/prepared/` with `seq`, `pssm`, `dssp`, `labels`,
`coords`, `train.txt`, `val.txt` and `test.txt`. These files are generated from
the bundled fused pickle files and are intentionally not versioned twice.

### Step 3: train and evaluate the full PertiNet-S model

```bash
python scripts/train_dset.py
```

The script selects the operating threshold on validation residues and applies
that frozen threshold once to the held-out test set. Results are written to
`results/dset/full_dset_results.json`, with the model checkpoint saved beside
the JSON file.

### Step 4: compare with the manuscript table

```bash
python scripts/compare_dset_results.py
```

The comparison reports the signed difference between the new run and the
manuscript values. Exact repetition can vary slightly with GPU libraries and
floating-point execution; the data membership and evaluation roles remain
fixed.

## RBP400 case-study workflow

RBP400 and RBP400-Expanded are not redistributed in this repository. Replace
every `...` below with the corresponding local path. RBP400-Expanded supplies
case-excluded development pairs; RBP400 supplies the HCC and lung candidate
universes.

Prepare case-safe development splits:

```bash
python experiments/rbp400/prepare_case_safe_pairs.py \
  --expanded-root ... \
  --rbp400-root ... \
  --cluster-id30 ... \
  --cluster-id20 ... \
  --output-root ...
```

Train the five-seed sequence ensemble using precomputed ESM embeddings:

```bash
python experiments/rbp400/train_esm_case_model.py \
  --split-root ... \
  --esm-cache ... \
  --output-root ... \
  --device cuda:0
```

Score the two candidate networks:

```bash
python experiments/rbp400/predict_hcc_lung_linear.py \
  --rbp400-root ... \
  --esm-cache ... \
  --model-root ... \
  --output-root ...
```

The RBP400 scripts do not use case-study candidate pairs for model fitting or
threshold selection. Machine-readable locked summaries are retained under
`results/rbp400/`.

## Data sources

- Bernett gold-standard dataset: https://doi.org/10.6084/m9.figshare.21591618
- RBP400 and RBP400-Expanded archive: https://doi.org/10.5281/zenodo.21320101
- Dset_186_72_PDB164 fused release: included under
  `data/Dset_186_72_PDB164/source/`

After downloading an external archive, replace the corresponding `...` in the
commands above with its local path. No author-specific server path is required.

## Code-to-manuscript map

| Manuscript component | Implementation |
|---|---|
| Multi-scale sequence encoder | `SequenceLocalEncoder` |
| Scalar/vector structural encoder | `GVPEncoder` |
| GO graph encoder | `GOFunctionEncoder` |
| Interaction-before-weighting token fusion | `CrossModalFusion` |
| Pair-level PertiNet output | `PertiNet` |
| Residue/interface output | `PertiNetS` |
| BCE, quadruplet and score-separation terms | `PertiNetObjective` |
| Validation-selected MCC threshold | `model/evaluation.py` |

## Repository layout

```text
data/bernett/                         fixed Bernett pair partitions
data/Dset_smoke_example/              three-record executable example
data/Dset_186_72_PDB164/source/       full fused Dset source and coordinates
examples/smoke_test.py                model-framework smoke test
examples/dset_smoke.py                real-data smoke test
model/                                PertiNet and PertiNet-S implementation
scripts/prepare_dset.py               fused-to-per-protein materialization
scripts/train_dset.py                 full Dset training and evaluation
experiments/rbp400/                   RBP400 case-study scripts
tests/test_release.py                 automated release tests
```

## License

See `LICENSE`.
