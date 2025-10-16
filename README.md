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

## Quickstart (inference -> Top-K -> plots)

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

