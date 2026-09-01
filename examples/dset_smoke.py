"""Run one real record from each Dset source subset through PertiNet-S."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import PertiNetS
from model.dset import load_prepared_record


EXAMPLE_ROOT = REPO_ROOT / "data" / "Dset_smoke_example"


def validate_dset_smoke():
    manifest = json.loads((EXAMPLE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    prepared = EXAMPLE_ROOT / "prepared"
    model = PertiNetS({
        "seq_input_dim": 40,
        "node_dims": (9, 1),
        "edge_dims": (1, 1),
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reports = []
    for item in manifest:
        record = load_prepared_record(prepared, item["record_id"])
        graph = record["graph"]
        logits = model({
            "seq_feat": record["seq_feat"],
            "seq_mask": record["seq_mask"],
            "x_s": graph.x_s,
            "x_v": graph.x_v,
            "edge_index": graph.edge_index,
            "edge_attr": graph.edge_attr,
        })
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, record["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reports.append({
            "record_id": item["record_id"],
            "source_subset": item["source_subset"],
            "residues": record["aligned_length"],
            "interface_residues": int(record["labels"].sum()),
            "ca_graph_edges": int(graph.edge_index.size(1)),
            "forward_backward": "passed",
        })
    return {
        "status": "passed",
        "purpose": "executable example; not a benchmark result",
        "records": len(reports),
        "source_subsets": sorted(item["source_subset"] for item in reports),
        "sequence_channels": 40,
        "dssp_scalar_channels": 9,
        "ca_cutoff_angstrom": 10.0,
        "details": reports,
    }


if __name__ == "__main__":
    print(json.dumps(validate_dset_smoke(), indent=2))
