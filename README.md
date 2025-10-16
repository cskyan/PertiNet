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

## 📦 Data & Weights

- Features (under data/RBP109/): sequence (one-hot/PSSM), structure graphs (GVP-ready), GO annotations & graph, and balanced PPI labels
- Weights (under weights/): e.g., fused109.best.pth
- Final CSV: weights/test_pairs_with_disturb_scores.csv produced by predict.py
If the repo is public, consider adding a tiny demo (10–20 pairs) or a short data pointers section so users can run end-to-end.
