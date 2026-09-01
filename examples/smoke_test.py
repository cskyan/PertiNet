"""Deterministic CPU smoke test for the manuscript PertiNet architecture.

This test uses synthetic tensors only.  It verifies imports, all three modality
encoders, the four fusion variants used in the ablation, forward/backward
execution, and the sample-wise modality-weight invariant.  It does not claim to
reproduce a benchmark metric.
"""

import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.data import Batch, Data

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import PertiNet, PertiNetS
from model.losses import PertiNetObjective


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def protein_graph(seed, n_nodes=5, node_scalar_dim=8, edge_scalar_dim=4):
    generator = torch.Generator().manual_seed(seed)
    src = torch.arange(n_nodes, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    n_edges = edge_index.size(1)
    return Data(
        num_nodes=n_nodes,
        x_s=torch.randn(n_nodes, node_scalar_dim, generator=generator),
        x_v=torch.randn(n_nodes, 1, 3, generator=generator),
        edge_index=edge_index,
        edge_attr=(
            torch.randn(n_edges, edge_scalar_dim, generator=generator),
            torch.zeros(n_edges, 0, 3),
        ),
    )


def make_batch(batch_size=2):
    graphs = []
    for pair_index in range(batch_size):
        graphs.extend([
            protein_graph(100 + pair_index * 2),
            protein_graph(101 + pair_index * 2),
        ])
    graph_batch = Batch.from_data_list(graphs)
    go_terms = 6
    go_edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 3], [1, 2, 3, 4, 5, 0, 0, 2]],
        dtype=torch.long,
    )
    return {
        "seq_feat": torch.randn(batch_size, 24, 40),
        "go_feat": torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=torch.float32),
        "go_edge_index": go_edge_index,
        "x_s": graph_batch.x_s,
        "x_v": graph_batch.x_v,
        "edge_index": graph_batch.edge_index,
        "edge_attr": graph_batch.edge_attr,
        "batch": graph_batch.batch,
    }, go_terms


def main():
    seed_everything()
    inputs, go_terms = make_batch()
    results = {}
    for fusion_type in ("token_transformer", "attention_only", "gated", "concat_mlp"):
        config = {
            "seq_input_dim": 40,
            "go_input_dim": 6,
            "num_go_terms": go_terms,
            "num_labels": 1,
            "mode": "full",
            "node_dims": (8, 1),
            "edge_dims": (4, 0),
            "fusion_type": fusion_type,
            "fusion_dropout": 0.0,
        }
        model = PertiNet(config)
        model.train()
        outputs = model(inputs)
        logits, margin, *_ = outputs
        objective = PertiNetObjective(pos_weight=1.0)
        loss, loss_parts = objective(outputs, torch.tensor([0.0, 1.0]))
        loss.backward()
        assert logits.shape == (2, 1)
        assert torch.isfinite(logits).all() and torch.isfinite(margin).all()
        weights = model.fusion.last_modality_weights
        if weights is not None:
            assert weights.shape == (2, 3)
            assert torch.allclose(weights.sum(1), torch.ones(2), atol=1e-6)
        results[fusion_type] = {
            "logit_shape": list(logits.shape),
            "margin_shape": list(margin.shape),
            "backward": "passed",
            "objective_terms": sorted(loss_parts),
            "modality_weight_sum": None if weights is None else [round(v, 6) for v in weights.sum(1).tolist()],
        }
    no_go_config = dict(config, fusion_type="token_transformer", use_go=False)
    no_go_model = PertiNet(no_go_config)
    no_go_model(inputs)
    no_go_weights = no_go_model.fusion.last_modality_weights
    assert torch.equal(no_go_weights[:, 2], torch.zeros_like(no_go_weights[:, 2]))
    graph = protein_graph(999)
    interface_model = PertiNetS({
        "seq_input_dim": 40, "node_dims": (8, 1), "edge_dims": (4, 0)
    })
    interface_logits = interface_model({
        "seq_feat": torch.randn(1, 5, 40),
        "seq_mask": torch.ones(1, 5, dtype=torch.bool),
        "x_s": graph.x_s,
        "x_v": graph.x_v,
        "edge_index": graph.edge_index,
        "edge_attr": graph.edge_attr,
    })
    assert interface_logits.shape == (5,)
    interface_logits.mean().backward()
    print(json.dumps({
        "status": "passed",
        "device": "cpu",
        "pair_variants": results,
        "interface_track": {"logit_shape": [5], "backward": "passed"},
    }, indent=2))


if __name__ == "__main__":
    main()
