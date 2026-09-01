# Full Dset_186_72_PDB164 release

This directory contains the complete 422-protein residue/interface benchmark
used by PertiNet-S. The repository distinguishes the versioned fused source from
the generated training representation.

## 1. Versioned fused source

`source/` is the input to `scripts/prepare_dset.py` and contains:

```text
source/
├── all_dset_list.pkl
├── fused_sequence_data.pkl
├── fused_pssm_data.pkl
├── fused_dssp_data.pkl
├── fused_label.pkl
├── fused_training_list.pkl   # 302 proteins
├── fused_validing_list.pkl   # 50 proteins
├── fused_test_list.pkl       # 70 proteins
├── deepppisp_split_manifest.json
└── coords/*.npz              # 422 real C-alpha coordinate records
```

The three split files are protein indices into the four fused feature/label
collections. They follow the released DeepPPISP 352-development/70-test
assignment. The 70 test proteins are retained exactly; 50 development proteins
are deterministically reserved for PertiNet validation with seed 2026, leaving
302 training proteins. The lists are disjoint and cover all 422 records.

## 2. Generated prepared representation

From the repository root, materialize the per-protein files:

```bash
python scripts/prepare_dset.py
```

This creates `prepared/seq`, `prepared/pssm`, `prepared/dssp`,
`prepared/labels`, `prepared/coords`, and the corresponding `train.txt`,
`val.txt`, and `test.txt`. These generated files are a direct expansion of the
versioned fused `pkl` files; they are not a second or independently sampled
dataset.

Run the full experiment with:

```bash
python scripts/train_dset.py
python scripts/compare_dset_results.py
```

No command-line data path is required when the repository layout is unchanged.
