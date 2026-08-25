# -*- coding: utf-8 -*-
"""Configuration entry point for PertiNet.

The dataclass lives in model.py so checkpoints can serialize a single
model-facing config object. This module provides a stable import path for
training/evaluation scripts.
"""

from model import TRIAGEConfig


CURRENT_DATA_ROOT = "data"


def default_config(**overrides) -> TRIAGEConfig:
    cfg = TRIAGEConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def current_data_config(**overrides) -> TRIAGEConfig:
    """Return the default multimodal training configuration."""
    cfg = TRIAGEConfig(
        project_root=CURRENT_DATA_ROOT,
        d_res_in=1309,
        d_model=128,
        batch_size=16,
        num_workers=8,
        topk=16,
        topm=64,
        pair_fourpack_dir=f"{CURRENT_DATA_ROOT}/TUnA/Intra1/fourpack",
        site_global_dir=f"{CURRENT_DATA_ROOT}/site_data/site_global",
        site_homo_dir=f"{CURRENT_DATA_ROOT}/site_data/HomoPDB_data",
        site_hetero_dir=f"{CURRENT_DATA_ROOT}/site_data/HeteroPDB_data",
        run_dir="outputs",
        use_pair_pssm=True,
        use_pair_dssp_ss=True,
        use_pair_dssp_rsa=True,
        use_site_pssm=True,
        use_site_dssp_ss=True,
        use_site_dssp_rsa=True,
        use_esm=True,
        pair_esm_dir=f"{CURRENT_DATA_ROOT}/TUnA/Intra1/fourpack/emb/esm2",
        site_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/site",
        use_coords_geometry=True,
        pair_rsa_available_rate=1.0,
        site_pssm_available_rate=0.9239130434782609,
        site_rsa_available_rate=1.0,
        val_fraction=0.10,
        test_fraction=0.10,
        max_pair_len=512,
        max_site_len=768,
        l1_lr=1e-4,
        l1_warmup_epochs=2,
        l1_scheduler_epochs=80,
        l1_min_lr_factor=0.30,
        epochs_l1_graphrbf=80,
        patience_l1_graphrbf=16,
        l1_batch_size=32,
        l1_num_workers=12,
        l1_pos_weight_cap=10.0,
        w_l1_rank=0.10,
        l1_rank_margin=0.20,
        l1_score_w_auc=0.35,
        l1_score_w_auprc=0.30,
        l1_score_w_mcc=0.20,
        l1_score_w_f1=0.15,
        l1_single_chain_mode=True,
        use_l1_raw_skip=True,
        l1_raw_skip_alpha=0.70,
        use_l1_multiscale_head=True,
        l1_multiscale_channels=64,
        l1_multiscale_delta_init=0.05,
        use_l1_geom_adapter=True,
        l1_geom_alpha=0.18,
        use_l1_geom_early=False,
        l1_geom_early_alpha=0.25,
        l1_threshold_beta=1.20,
        pp_root=f"{CURRENT_DATA_ROOT}/Dest_prepared",
        pp_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/dest",
        rbp400_root=f"{CURRENT_DATA_ROOT}/RBP400",
        rbp400_id_list=f"{CURRENT_DATA_ROOT}/RBP400/accessions.txt",
        rbp400_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/rbp400",
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def topk_engineering_data_config(**overrides) -> TRIAGEConfig:
    """Return the top-k pretraining configuration."""
    cfg = current_data_config(
        pair_fourpack_dir=f"{CURRENT_DATA_ROOT}/TUnA/Intra1/fourpack",
        site_global_dir=f"{CURRENT_DATA_ROOT}/site_data/site_global",
        site_homo_dir=f"{CURRENT_DATA_ROOT}/site_data/HomoPDB_data",
        site_hetero_dir=f"{CURRENT_DATA_ROOT}/site_data/HeteroPDB_data",
        use_pair_pssm=True,
        use_pair_dssp_ss=True,
        use_pair_dssp_rsa=False,
        use_site_pssm=True,
        use_site_dssp_ss=True,
        use_site_dssp_rsa=True,
        use_coords_geometry=True,
        pair_rsa_available_rate=0.0145513338722716,
        site_pssm_available_rate=0.9239130434782609,
        site_rsa_available_rate=1.0,
        val_fraction=0.10,
        max_pair_len=512,
        max_site_len=768,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def rbp400_data_config(**overrides) -> TRIAGEConfig:
    """RBP400 single-chain L1 config on top of the TRIAGE engineering model.

    Keep the engineering backbone/pretrained checkpoint and the L1-specific
    evidence heads that gave the stronger RBP400 runs. Rank/top-k auxiliary
    losses stay off for the binary run, but the architecture should not be
    collapsed to the plain PP head.
    """
    cfg = current_data_config(
        rbp400_root=f"{CURRENT_DATA_ROOT}/RBP400",
        rbp400_id_list=f"{CURRENT_DATA_ROOT}/RBP400/accessions.txt",
        rbp400_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/rbp400",
        pp_root=f"{CURRENT_DATA_ROOT}/RBP400",
        pp_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/rbp400",
        dropout=0.10,
        l1_lr=5e-5,
        l1_warmup_epochs=2,
        l1_min_lr_factor=0.30,
        l1_ema_decay=0.0,
        epochs_l1_graphrbf=80,
        patience_l1_graphrbf=16,
        l1_batch_size=16,
        l1_num_workers=8,
        max_site_len=1024,
        l1_pos_weight_cap=10.0,
        l1_label_smoothing=0.0,
        w_l1_rank=0.03,
        l1_rank_margin=0.20,
        l1_rank_start_epoch=8,
        l1_rank_ramp_epochs=5,
        l1_rank_max_pairs=2048,
        l1_per_protein_loss=False,
        l1_extreme_label_weight=1.0,
        w_l1_hard_rank=0.0,
        l1_hard_rank_margin=0.25,
        l1_hard_rank_neg_frac=0.03,
        l1_hard_rank_max_neg=12,
        l1_hard_rank_start_epoch=18,
        l1_hard_rank_ramp_epochs=8,
        w_l1_topband_bce=0.0,
        l1_topband_frac=0.20,
        l1_topband_min_k=10,
        l1_topband_max_k=128,
        l1_topband_start_epoch=999,
        l1_topband_ramp_epochs=1,
        w_l1_l10_boundary=0.0,
        l1_l10_boundary_frac=0.10,
        l1_l10_boundary_margin=0.04,
        l1_l10_boundary_max_pos=64,
        l1_l10_boundary_start_epoch=9,
        l1_l10_boundary_ramp_epochs=6,
        l1_threshold_mode="auto_acc",
        l1_threshold_min_recall=0.0,
        l1_threshold_beta=1.0,
        l1_score_w_auc=0.15,
        l1_score_w_auprc=0.10,
        l1_score_w_mcc=0.25,
        l1_score_w_f1=0.05,
        l1_score_w_acc=0.45,
        l1_score_loss_penalty=0.0,
        l1_score_w_recall_l5=0.45,
        l1_score_w_recall_l10=0.35,
        l1_score_w_precision_10=0.10,
        l1_score_w_hit_2=0.0,
        l1_score_w_hit_20=0.10,
        l1_ager_enable=False,
        l1_ager_radius=10.0,
        l1_ager_alpha=0.20,
        l1_ager_top_m=5,
        use_l1_raw_skip=True,
        use_l1_multiscale_head=True,
        use_l1_geom_adapter=True,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def dest_data_config(**overrides) -> TRIAGEConfig:
    """Dest/Dset L1 fine-tune config on top of the engineering TRIAGE model.

    The raw GraphRBF-style data live under ``the raw Dest directory``. Run
    ``prepare_dest_from_graphrbf_pkl.py`` once to materialize the prepared
    directory used by ``GraphRBFPPDataset``.
    """
    cfg = current_data_config(
        run_dir=f"{CURRENT_DATA_ROOT}/triage_runs/dest_from_engineering",
        dest_raw_root=f"{CURRENT_DATA_ROOT}/Dset",
        dest_root=f"{CURRENT_DATA_ROOT}/Dset_prepared",
        dest_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/dest",
        dest_base_checkpoint=f"{CURRENT_DATA_ROOT}/triage_runs/triage_final.pt",
        pp_root=f"{CURRENT_DATA_ROOT}/Dset_prepared",
        pp_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/dest",
        dropout=0.10,
        l1_lr=1e-4,
        l1_warmup_epochs=2,
        l1_min_lr_factor=0.30,
        l1_ema_decay=0.0,
        epochs_l1_graphrbf=80,
        patience_l1_graphrbf=16,
        l1_batch_size=32,
        l1_num_workers=12,
        max_site_len=1024,
        l1_pos_weight_cap=10.0,
        l1_label_smoothing=0.0,
        w_l1_rank=0.12,
        l1_rank_margin=0.20,
        l1_rank_start_epoch=1,
        l1_rank_ramp_epochs=3,
        l1_rank_max_pairs=4096,
        w_l1_hard_rank=0.02,
        l1_hard_rank_margin=0.25,
        l1_hard_rank_neg_frac=0.05,
        l1_hard_rank_max_neg=24,
        l1_hard_rank_start_epoch=8,
        l1_hard_rank_ramp_epochs=6,
        w_l1_topband_bce=0.02,
        l1_topband_frac=0.20,
        l1_topband_start_epoch=6,
        l1_topband_ramp_epochs=4,
        w_l1_l10_boundary=0.02,
        l1_l10_boundary_start_epoch=10,
        l1_l10_boundary_ramp_epochs=6,
        l1_threshold_mode="auto_mcc",
        l1_threshold_min_recall=0.0,
        l1_threshold_beta=1.20,
        l1_score_w_auc=0.25,
        l1_score_w_auprc=0.25,
        l1_score_w_mcc=0.15,
        l1_score_w_f1=0.05,
        l1_score_w_recall_l5=0.15,
        l1_score_w_recall_l10=0.10,
        l1_score_w_precision_10=0.05,
        l1_score_w_acc=0.0,
        l1_score_loss_penalty=0.0,
        l1_single_chain_mode=True,
        use_l1_raw_skip=True,
        l1_raw_skip_alpha=0.70,
        use_l1_multiscale_head=True,
        l1_multiscale_channels=64,
        use_l1_geom_adapter=True,
        l1_geom_alpha=0.20,
        use_l1_geom_early=False,
        use_esm=True,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def dest_triage_config(**overrides) -> TRIAGEConfig:
    """Dest weak tri-level fusion fine-tune.

    Dest is residue-level binary PPIS data, not a pair-level benchmark with
    negative pairs or true contact maps. This config therefore keeps residue
    binary classification as the primary objective while enabling weak
    L2/L3 fusion through same-PDB chain pairs.
    """
    cfg = dest_data_config(
        run_dir=f"{CURRENT_DATA_ROOT}/triage_runs/dest_triage_from_engineering",
        dest_base_checkpoint=f"{CURRENT_DATA_ROOT}/triage_runs/triage_final.pt",
        batch_size=8,
        l1_batch_size=8,
        l1_num_workers=8,
        dest_pairing_mode="same_pdb_or_self",
        dest_pair_max_partners=1,
        epochs_l1_graphrbf=80,
        patience_l1_graphrbf=14,
        l1_lr=6e-5,
        w_res=1.0,
        dest_balanced_res_loss=True,
        w_contact=0.06,
        w_l1_l2_consistency=0.04,
        w_topk_contact_ranking=0.0,
        w_topk_margin_rank=0.0,
        w_triage_struct=0.03,
        w_struct_gate_regularization=0.02,
        struct_max_pair_gate=0.55,
        struct_min_local_gate=0.30,
        l1_threshold_mode="auto_mcc",
        l1_threshold_beta=1.20,
        l1_score_w_acc=0.10,
        l1_score_w_auc=0.30,
        l1_score_w_auprc=0.25,
        l1_score_w_mcc=0.25,
        l1_score_w_f1=0.10,
        l1_score_w_recall_l5=0.0,
        l1_score_w_recall_l10=0.0,
        l1_score_w_precision_10=0.0,
        w_l1_rank=0.0,
        w_l1_hard_rank=0.0,
        w_l1_topband_bce=0.0,
        w_l1_l10_boundary=0.0,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def rbp400_topk_data_config(**overrides) -> TRIAGEConfig:
    """RBP400 top-k L1 config on top of the TRIAGE engineering model.

    This keeps the same engineering backbone path as the binary RBP400 run,
    but checkpoint selection and auxiliary losses follow the older top-k
    protocol: Recall@L/5, Recall@L/10, Precision@10, and Hit@20.
    """
    cfg = current_data_config(
        rbp400_root=f"{CURRENT_DATA_ROOT}/RBP400",
        rbp400_id_list=f"{CURRENT_DATA_ROOT}/RBP400/accessions.txt",
        rbp400_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/rbp400",
        pp_root=f"{CURRENT_DATA_ROOT}/RBP400",
        pp_esm_dir=f"{CURRENT_DATA_ROOT}/esm_embeddings/rbp400",
        dropout=0.10,
        l1_lr=1e-4,
        l1_warmup_epochs=2,
        l1_min_lr_factor=0.30,
        l1_ema_decay=0.0,
        epochs_l1_graphrbf=80,
        patience_l1_graphrbf=16,
        l1_batch_size=16,
        l1_num_workers=8,
        max_site_len=1024,
        l1_pos_weight_cap=10.0,
        l1_label_smoothing=0.0,
        w_l1_rank=0.10,
        l1_rank_margin=0.20,
        l1_rank_start_epoch=1,
        l1_rank_ramp_epochs=1,
        l1_rank_max_pairs=2048,
        l1_per_protein_loss=True,
        l1_extreme_label_weight=0.35,
        w_l1_hard_rank=0.0,
        l1_hard_rank_margin=0.25,
        l1_hard_rank_neg_frac=0.03,
        l1_hard_rank_max_neg=12,
        l1_hard_rank_start_epoch=18,
        l1_hard_rank_ramp_epochs=8,
        w_l1_topband_bce=0.0,
        l1_topband_frac=0.20,
        l1_topband_min_k=10,
        l1_topband_max_k=128,
        l1_topband_start_epoch=999,
        l1_topband_ramp_epochs=1,
        w_l1_l10_boundary=0.025,
        l1_l10_boundary_frac=0.10,
        l1_l10_boundary_margin=0.04,
        l1_l10_boundary_max_pos=64,
        l1_l10_boundary_start_epoch=9,
        l1_l10_boundary_ramp_epochs=6,
        l1_threshold_mode="auto_mcc",
        l1_threshold_min_recall=0.0,
        l1_threshold_beta=1.0,
        l1_score_w_auc=0.0,
        l1_score_w_auprc=0.0,
        l1_score_w_mcc=0.0,
        l1_score_w_f1=0.0,
        l1_score_w_acc=0.0,
        l1_score_loss_penalty=0.0,
        l1_score_w_recall_l5=0.45,
        l1_score_w_recall_l10=0.35,
        l1_score_w_precision_10=0.10,
        l1_score_w_hit_2=0.0,
        l1_score_w_hit_20=0.10,
        l1_ager_enable=True,
        l1_ager_radius=10.0,
        l1_ager_alpha=0.20,
        l1_ager_top_m=5,
        use_l1_raw_skip=True,
        use_l1_multiscale_head=True,
        use_l1_geom_adapter=True,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def rbp400_triage_data_config(**overrides) -> TRIAGEConfig:
    """RBP400 tri-level fine-tune config for the case-study model.

    This stage starts from ``triage_topk_final.pt`` and trains the RBP400 L1
    residue-ranking branch together with the engineering pair/interface
    branches. Keep run-time paths and batch sizes here so the CLI stays focused
    on run control rather than experiment wiring.
    """
    pair_cfg = topk_engineering_data_config()
    cfg = rbp400_topk_data_config(
        batch_size=2,
        l1_batch_size=4,
        l1_lr=2e-5,
        w_l1_rank=0.12,
        w_l1_l10_boundary=0.04,
        l1_l10_boundary_start_epoch=1,
        l1_l10_boundary_ramp_epochs=3,
        l1_score_w_recall_l5=0.30,
        l1_score_w_recall_l10=0.30,
        l1_score_w_precision_10=0.25,
        l1_score_w_hit_20=0.15,
        rbp400_triage_score_w_pair_auprc=0.05,
        rbp400_triage_score_w_pair_mcc=0.03,
        rbp400_triage_score_w_recall_l5=0.30,
        rbp400_triage_score_w_recall_l10=0.30,
        rbp400_triage_score_w_precision_10=0.25,
        rbp400_triage_score_w_hit_20=0.05,
        rbp400_triage_score_w_gate_entropy=0.02,
        rbp400_triage_score_loss_penalty=0.01,
        pair_fourpack_dir=pair_cfg.pair_fourpack_dir,
        pair_esm_dir=pair_cfg.pair_esm_dir,
        use_pair_pssm=pair_cfg.use_pair_pssm,
        use_pair_dssp_ss=pair_cfg.use_pair_dssp_ss,
        use_pair_dssp_rsa=pair_cfg.use_pair_dssp_rsa,
        pair_rsa_available_rate=pair_cfg.pair_rsa_available_rate,
    )
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def tuna_pair_finetune_config(**overrides) -> TRIAGEConfig:
    """TUnA pair-level fine-tune config on top of the engineering model.

    This is the checkpoint that should be used for the TUnA benchmark and the
    fusion/calibration/gate diagnostics. Runtime commands should not need to
    repeat paths or ordinary hyperparameters; keep those choices here.
    """
    cfg = current_data_config(
        run_dir=f"{CURRENT_DATA_ROOT}/triage_runs/tuna_from_engineering",
        pair_fourpack_dir=f"{CURRENT_DATA_ROOT}/TUnA/Intra1/fourpack",
        pair_esm_dir=f"{CURRENT_DATA_ROOT}/TUnA/Intra1/fourpack/emb/esm2",
        use_pair_pssm=True,
        use_pair_dssp_ss=True,
        use_pair_dssp_rsa=True,
        pair_rsa_available_rate=1.0,
        batch_size=16,
        num_workers=8,
        val_fraction=0.10,
        test_fraction=0.10,
        max_pair_len=512,
        epochs_pair_fusion=30,
        patience_pair_fusion=8,
        lr=1e-4,
        weight_decay=1e-4,
    )
    cfg.tuna_base_checkpoint = f"{CURRENT_DATA_ROOT}/triage_runs/triage_final.pt"
    cfg.tuna_split_mode = "protein_component"
    cfg.tuna_seed = 42
    cfg.tuna_checkpoint_name = "triage_tuna_pair_finetuned.pt"
    cfg.tuna_diagnostics_dir = f"{CURRENT_DATA_ROOT}/triage_if_extra/tuna_fusion_analysis"
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown TRIAGEConfig field: {key}")
        setattr(cfg, key, value)
    return cfg


def feature_spec(cfg: TRIAGEConfig) -> dict:
    return {
        "external_annotation_input": False,
        "pair": {
            "sequence_embedding": True,
            "esm_dir": cfg.pair_esm_dir if bool(getattr(cfg, "use_esm", True)) else "",
            "pssm": bool(cfg.use_pair_pssm),
            "dssp_ss": bool(cfg.use_pair_dssp_ss),
            "dssp_rsa": bool(cfg.use_pair_dssp_rsa),
            "dssp_rsa_available_rate": float(cfg.pair_rsa_available_rate),
        },
        "site_struct": {
            "sequence_embedding": True,
            "esm_dir": cfg.site_esm_dir if bool(getattr(cfg, "use_esm", True)) else "",
            "pssm": bool(cfg.use_site_pssm),
            "pssm_available_rate": float(cfg.site_pssm_available_rate),
            "dssp_ss": bool(cfg.use_site_dssp_ss),
            "dssp_rsa": bool(cfg.use_site_dssp_rsa),
            "dssp_rsa_available_rate": float(cfg.site_rsa_available_rate),
            "coords_geometry": bool(cfg.use_coords_geometry),
        },
    }
