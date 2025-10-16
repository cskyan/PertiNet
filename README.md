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

## Data download and preparation
This project uses two kinds of data:
1) the multimodal RBP109 feature pack used by the model (sequence / structure / GO);
2) optional external resources for the ILF3/PTBP1 case study (TCGA-LIHC expression, STRING v12 network, TTD targets).

Below is a minimal, reproducible way to fetch/build what the repo expects on disk.

### A. RBP109 multimodal features (required)

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
