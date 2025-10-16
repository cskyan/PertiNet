# PertiNet

PertiNet is a multimodal deep learning framework for Protein–Protein Interaction (PPI) prediction with a learned threshold and a disturbance score for condition-specific analysis.
This repo includes training/inference scripts, pre-trained weights for the RBP109 study, plotting utilities, and an ILF3/PTBP1 case workflow.

✨ Highlights

Three modalities: sequence (CNN+Transformer), structure (GVP-GNN), function (GO graph via GATv2)

Fusion: transformer-based feature fusion for robust PPI prediction

Outputs: PPI score s, learned threshold τ, and disturbance score d = |s − τ|

Repro kit: pre-trained weights, final prediction CSV, Top-K extractor, and figure scripts
