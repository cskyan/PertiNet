"""Default release paths for the manuscript experiments."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# Fused Dset_186/Dset_72/PDBset_164 source used for the manuscript experiment.
# PERTINET_DSET_ROOT may be set when the repository is installed elsewhere.
DSET_ROOT = Path(
    os.environ.get(
        "PERTINET_DSET_ROOT",
        str(REPO_ROOT / "data" / "Dset_186_72_PDB164" / "source"),
    )
)

FUSED_DSET_FILES = {
    "sequence": "fused_sequence_data.pkl",
    "pssm": "fused_pssm_data.pkl",
    "dssp": "fused_dssp_data.pkl",
    "label": "fused_label.pkl",
    "train": "fused_training_list.pkl",
    "validation": "fused_validing_list.pkl",
    "test": "fused_test_list.pkl",
}
