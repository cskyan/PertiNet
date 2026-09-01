"""Prepared Dset residue/interface input loader and C-alpha graph builder."""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from .config import DSET_ROOT, FUSED_DSET_FILES


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}


def load_fused_dset(root=DSET_ROOT):
    """Load the fused manuscript Dset arrays and fixed split indices."""
    root = Path(root)
    missing = [name for name in FUSED_DSET_FILES.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing fused Dset files under {root}: {missing}")
    bundle = {}
    for key, filename in FUSED_DSET_FILES.items():
        with (root / filename).open("rb") as handle:
            bundle[key] = pickle.load(handle)
    n_records = len(bundle["sequence"])
    for key in ("pssm", "dssp", "label"):
        if len(bundle[key]) != n_records:
            raise ValueError(
                f"fused {key} contains {len(bundle[key])} records; expected {n_records}"
            )
    for split in ("train", "validation", "test"):
        indices = [int(index) for index in bundle[split]]
        if any(index < 0 or index >= n_records for index in indices):
            raise IndexError(f"{split} contains an out-of-range fused record index")
        bundle[split] = indices
    bundle["root"] = root
    bundle["n_records"] = n_records
    return bundle


def read_fasta(path):
    sequence = "".join(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    ).upper()
    if not sequence:
        raise ValueError(f"empty FASTA: {path}")
    return sequence


def build_ca_graph(coords, dssp, cutoff=10.0):
    """Build the manuscript C-alpha graph with scalar/vector GVP features."""
    coords = torch.as_tensor(coords, dtype=torch.float32)
    dssp = torch.as_tensor(dssp, dtype=torch.float32)
    if coords.ndim != 2 or coords.size(1) != 3 or dssp.ndim != 2:
        raise ValueError("coords must be [N,3] and DSSP must be [N,D]")
    if coords.size(0) != dssp.size(0):
        raise ValueError("coordinate and DSSP residue counts must align")
    distances = torch.cdist(coords, coords)
    mask = (distances <= float(cutoff)) & (distances > 0)
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    delta = coords[dst] - coords[src]
    edge_distance = distances[src, dst].unsqueeze(-1)
    edge_vector = (delta / edge_distance.clamp_min(1e-6)).unsqueeze(1)
    centered = coords - coords.mean(dim=0, keepdim=True)
    node_vector = (
        centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(1e-6)
    ).unsqueeze(1)
    return Data(
        num_nodes=coords.size(0),
        x_s=dssp,
        x_v=node_vector,
        edge_index=edge_index,
        edge_attr=(edge_distance / float(cutoff), edge_vector),
    )


def load_prepared_record(root, record_id, cutoff=10.0):
    """Load one prepared record into PertiNet-S tensors."""
    root = Path(root)
    sequence = read_fasta(root / "seq" / f"{record_id}.fasta")
    pssm = np.load(root / "pssm" / f"{record_id}.npy", allow_pickle=False)
    dssp = np.load(root / "dssp" / f"{record_id}.npy", allow_pickle=False)
    labels = np.load(root / "labels" / f"{record_id}.npy", allow_pickle=False)
    with np.load(root / "coords" / f"{record_id}.npz", allow_pickle=False) as archive:
        coords = archive["coords"]
    length = min(len(sequence), len(pssm), len(dssp), len(labels), len(coords))
    if length == 0:
        raise ValueError(f"empty aligned record: {record_id}")
    aa_indices = np.array([AA_TO_INDEX.get(aa, 0) for aa in sequence[:length]], dtype=np.int64)
    one_hot = np.eye(20, dtype=np.float32)[aa_indices]
    seq_features = np.concatenate([one_hot, np.asarray(pssm[:length], dtype=np.float32)], axis=1)
    graph = build_ca_graph(coords[:length], dssp[:length], cutoff=cutoff)
    return {
        "record_id": record_id,
        "seq_feat": torch.from_numpy(seq_features).unsqueeze(0),
        "seq_mask": torch.ones(1, length, dtype=torch.bool),
        "graph": graph,
        "labels": torch.as_tensor(labels[:length], dtype=torch.float32),
        "aligned_length": length,
        "original_lengths": {
            "sequence": len(sequence), "pssm": len(pssm), "dssp": len(dssp),
            "labels": len(labels), "coords": len(coords),
        },
    }
