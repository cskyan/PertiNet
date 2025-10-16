# PertiNet — Multimodal PPI Prediction & Disturbance Scoring

PertiNet is a multimodal deep learning framework for Protein–Protein Interaction (PPI) prediction with a learned threshold and a disturbance score for condition-specific analysis.
This repo includes training/inference scripts, pre-trained weights for the RBP109 study, a Top-K extractor, and plotting utilities for the ILF3/PTBP1 case.

---

## ✨Highlights

- Three modalities: sequence (CNN+Transformer), structure (GVP-GNN), function (GO graph via GATv2)
- Fusion: transformer-based feature fusion for robust PPI prediction
- Outputs: PPI score s, learned threshold τ, and disturbance score d = |s − τ|
- Repro kit: pre-trained weights, final prediction CSV, Top-K extractor, and figure scripts

---

## 📁Repository Layout 
```text
├─ data/
│  └─ RBP109/
│     ├─ all_go_annotations.tsv
│     ├─ dssp_109.npz
│     ├─ go_multi_hot_109.npy
│     ├─ go_term_edge_index.npy
│     ├─ human_reviewed_uniprot_ids.txt
│     ├─ ppi_labels_balanced.csv
│     ├─ ppi_labels_balanced.npy
│     ├─ protein_graphs_109.pkl
│     ├─ pssm_109.npz
│     └─ sequence_onehot.npy
├─ model/
│  ├─ gvp/
│  ├─ make_ilf3_ptbp1_topk.py
│  ├─ model.py
│  ├─ picture.py
│  ├─ predict.py
│  └─ train.py
├─ weights/
│  ├─ fused.best.pth
│  └─ fused109.best.pth
├─ LICENSE
└─ README.md
```
---

## 📥Data download and preparation
This project uses **two** datasets only:
1) **RBP109** multimodal features (sequence / structure / GO) — **already included** in this repo.
2) **Public Benchmark Dataset: Dset_186_72_PDB164** — can be reconstructed from the Protein Data Bank (PDB): https://www.rcsb.org/

Below is a minimal, reproducible way to fetch/build what the repo expects on disk.

### A. RBP109 multimodal features (included)

**What we need under `data/RBP109/`:**
```text
all_go_annotations.tsv
dssp_109.npz
go_multi_hot_109.npy
go_term_edge_index.npy
human_reviewed_uniprot_ids.txt
ppi_labels_balanced.csv
ppi_labels_balanced.npy
protein_graphs_109.pkl
pssm_109.npz
sequence_onehot.npy
```
No extra download is required.

### B. Public Benchmark Dataset: Dset_186_72_PDB164 (from PDB)
All entries are retrieved from the Protein Data Bank (PDB): https://www.rcsb.org/

**What it is.** A unified benchmark integrating three widely used subsets, all derived from PDB:
- **Dset_186**: 186 protein chains, filtered to < 25% sequence identity and resolution ≤ 3.0 Å, focusing on high-quality structure-based PPI prediction.
- **Dset_72**: 72 non-redundant protein structures with clearly annotated interface residues, complementing Dset_186 with broader structural diversity.
- **PDBset_164**: 164 curated protein chains chosen for functional diversity and validated binding-interface annotations.

**How it is built (summary).**
- Merge Dset_186, Dset_72, and PDBset_164; remove redundant chains by sequence similarity.
- Keep chains with **sequence identity < 25%** and **resolution ≤ 3.0 Å** to ensure quality and minimize redundancy.
- Standardize each chain to extract amino-acid sequences, 3D structural features (e.g., DSSP-derived), and graph representations (e.g., GVP encoders).
- Define residue-level interface labels by atom-distance: a residue is an interface site if any heavy atom is within **6.0 Å** of any atom from another chain; otherwise, non-binding.
- Split into **train/val/test = 8:1:1** with **pair-level low-homology control**: no homologous pairs (≥ 20% sequence identity for both proteins) appear across different partitions.

**Where to get it.**
- Download source structures directly from PDB (https://www.rcsb.org/) and reconstruct using the criteria above.
- If you already maintain your own copy of Dset_186, Dset_72, and PDBset_164, you can merge and filter them as specified to obtain Dset_186_72_PDB164.

Fetch PDB/mmCIF files for each protein into scratch/pdb/ (choose representative entries), then:
## 🚀Quickstart (inference -> Top-K -> plots)

Prerequisites: Python 3.7/3.8; CUDA 11.6+ recommended for GPU.  
Note: some paths in scripts are placeholders by design. Pass arguments or edit as needed.

### 1) Run inference to produce the final CSV
```bash
# from repo root
cd model

python predict.py --base_dir "../data/RBP109" \
                  --model_dir "../weights"

# Output file:
#   ../weights/test_pairs_with_disturb_scores.csv
# Columns include: Protein_A, Protein_B, pred_score, disturb_score, label, ...
```
### 2) Extract ILF3/PTBP1 Top-K files for plotting
```bash
# Default centers are ILF3=Q12906, PTBP1=P26599; default K = 50,30,10
python make_ilf3_ptbp1_topk.py --model_dir "D:\path\to\pertinet_model"

# Custom K / centers:
python make_ilf3_ptbp1_topk.py --model_dir "D:\path\to\pertinet_model" \
                               --k 100 50 20 --centers Q12906 P26599
# This produces, under {model_dir}:
#   top50_ILF3_PTBP1_disturb.csv
#   top30_ILF3_PTBP1_disturb.csv
#   top10_ILF3_PTBP1_disturb.csv
#   top_ILF3_PTBP1_nodes_50.txt (and for 30 / 10)
```

### 3) Generate figures
Open picture.py and set:
```bash
picture_dir = r"../fig_output"     # where figures will be saved
model_dir   = r"../weights"        # same directory used in step (2)
```
Then：
```bash
# still under model/
python picture.py
```
Figures will be saved into picture_dir.

## 🧪Training (optional)
```bash
# from repo root
cd model
python train.py --base_dir "../data/RBP109" \
                --model_dir "../weights"
```
Paths in scripts may contain placeholders by design; pass arguments or edit as needed.

## 📦 Weights

- Features (under data/RBP109/): sequence (one-hot/PSSM), structure graphs (GVP-ready), GO annotations & graph, and balanced PPI labels
- Weights (under weights/): e.g., fused109.best.pth
- Final CSV: weights/test_pairs_with_disturb_scores.csv produced by predict.py

## 🔧 Key Libraries & Versions (from the provided env)
These are the relevant packages for running this repo; others in your env are optional.
### Core DL
- torch==1.13.1+cu116, torchvision==0.14.1+cu116, torchaudio==0.13.1+cu116
- torch-geometric==2.3.1
- torch-scatter==2.1.1, torch-sparse==0.6.17, torch-cluster==1.6.1, torch-spline-conv==1.2.2
- pytorch-lightning==1.9.5 
- triton==1.0.0

### Protein/Sequence & Graph
- fair-esm==2.0.0 
- biopython==1.81
- networkx==2.6.3
- ogb==1.3.6
- goatools==1.4.12

### Data / Metrics / Plotting
- pandas 1.1.5
- numpy 1.21.6
- scikit-learn 1.0.2
- matplotlib 3.5.3
- seaborn 0.12.2
- gseapy 1.1.9
