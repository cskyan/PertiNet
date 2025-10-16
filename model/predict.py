# predict.py

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import (
    f1_score, average_precision_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score,
)
from scipy.special import expit
import matplotlib.pyplot as plt


# ===== Paths (EDIT ONLY THESE TWO) =====
base_dir = "..."   # was: /srv/storage/.../RBP109_raw_features
model_dir = "..."  # was: /srv/storage/.../model
os.makedirs(model_dir, exist_ok=True)

# keep file names identical to your original code
model_save_path        = os.path.join(model_dir, "fused109.best.pth")
sequence_path          = os.path.join(base_dir, "sequence_onehot.npy")
pssm_path              = os.path.join(base_dir, "pssm_109.npz")
dssp_path              = os.path.join(base_dir, "dssp_109.npz")
go_features_path       = os.path.join(base_dir, "go_multi_hot_109.npy")
go_edge_index_path     = os.path.join(base_dir, "go_term_edge_index.npy")
structure_graph_path   = os.path.join(base_dir, "protein_graphs_109.pkl")
test_pairs_csv_in_model= os.path.join(model_dir, "test_pairs109.csv")  # preserved name


# ===== Small helpers =====
def ensure_2d(arr, target_dim=None):
    """Ensure arr is (N, D); pad/truncate to target_dim if provided."""
    arr = np.array(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]
    if target_dim is None:
        return arr
    if arr.shape[1] < target_dim:
        arr = np.pad(arr, ((0, 0), (0, target_dim - arr.shape[1])), mode="constant")
    elif arr.shape[1] > target_dim:
        arr = arr[:, :target_dim]
    return arr


def to_gvp_vector(arr):
    """
    Return vector features with shape (N, 1, 3).
    If not available, return zeros of that shape.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:, None, :]
    N = arr.shape[0]
    return np.zeros((N, 1, 3), dtype=np.float32)


def to_gvp_edge_attr(edge_attr, target_dim):
    """Edge attributes as (edge_scalar [E, S], edge_vector [E, 0, 3])."""
    edge_scalar = ensure_2d(edge_attr, target_dim=target_dim)
    num_edges = edge_scalar.shape[0]
    edge_vector = np.zeros((num_edges, 0, 3), dtype=np.float32)
    return (
        torch.tensor(edge_scalar, dtype=torch.float32),
        torch.tensor(edge_vector, dtype=torch.float32),
    )


# ===== Dataset =====
class PPIPairDataset(Dataset):
    """
    Each item returns:
      x = {
        "seq_feat": [L, D_seq_pssm_dssp],
        "go_feat":  [G],       # multi-hot for pair (concatenated per your pipeline)
        "struct":   [Data_i, Data_j],  # two protein graphs
      }
      y = [label]
    """
    def __init__(self, seq_data, pssm_data, dssp_data, go_data, struct_data, label_data, max_len=500):
        self.seq_data = seq_data
        self.pssm_data = pssm_data
        self.dssp_data = dssp_data
        self.go_data = go_data
        self.struct_data = struct_data
        self.labels = label_data
        self.max_len = max_len
        assert len(self.seq_data) == len(self.labels), "Inconsistent sample sizes."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = np.array(self.seq_data[idx])
        pssm = np.array(self.pssm_data[idx])
        dssp = np.array(self.dssp_data[idx])
        go   = np.array(self.go_data[idx])
        label = float(self.labels[idx])

        # build two structure graphs
        g_i, g_j = self.struct_data[idx][0], self.struct_data[idx][1]
        x_s_dim = max(np.array(g_i["x_s"]).shape[1], np.array(g_j["x_s"]).shape[1])
        e_dim   = max(np.array(g_i["edge_attr"]).shape[1], np.array(g_j["edge_attr"]).shape[1])

        es_i, ev_i = to_gvp_edge_attr(g_i["edge_attr"], target_dim=e_dim)
        data_i = Data(
            x_s=torch.tensor(ensure_2d(g_i["x_s"], target_dim=x_s_dim), dtype=torch.float32),
            x_v=torch.tensor(to_gvp_vector(g_i["x_v"]), dtype=torch.float32),
            edge_index=torch.tensor(g_i["edge_index"], dtype=torch.long),
            edge_attr=(es_i, ev_i),
        )

        es_j, ev_j = to_gvp_edge_attr(g_j["edge_attr"], target_dim=e_dim)
        data_j = Data(
            x_s=torch.tensor(ensure_2d(g_j["x_s"], target_dim=x_s_dim), dtype=torch.float32),
            x_v=torch.tensor(to_gvp_vector(g_j["x_v"]), dtype=torch.float32),
            edge_index=torch.tensor(g_j["edge_index"], dtype=torch.long),
            edge_attr=(es_j, ev_j),
        )

        # concatenate sequence-level features along length, then pad/crop
        min_len = min(len(seq), len(pssm), len(dssp))
        x_raw = np.concatenate([seq[:min_len], pssm[:min_len], dssp[:min_len]], axis=-1)
        x_raw = x_raw[:self.max_len]
        if x_raw.shape[0] < self.max_len:
            pad_len = self.max_len - x_raw.shape[0]
            x_raw = np.pad(x_raw, ((0, pad_len), (0, 0)), mode="constant")

        return {
            "seq_feat": torch.tensor(x_raw, dtype=torch.float32),
            "go_feat":  torch.tensor(go, dtype=torch.float32),
            "struct":   [data_i, data_j],
        }, torch.tensor([label], dtype=torch.float32)


# ===== Main predict() =====
def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load resources
    with open(os.path.join(base_dir, "human_reviewed_uniprot_ids.txt")) as f:
        ids = [x.strip() for x in f if x.strip()]

    test_pairs = pd.read_csv(test_pairs_csv_in_model)

    with open(structure_graph_path, "rb") as f:
        structure_graphs = pickle.load(f)

    sequence_data = np.load(sequence_path, allow_pickle=True)
    pssm_data     = np.load(pssm_path, allow_pickle=True)   # dict-like .npz
    dssp_data     = np.load(dssp_path, allow_pickle=True)   # dict-like .npz
    go_features   = np.load(go_features_path)
    go_edge_index_tensor = torch.tensor(np.load(go_edge_index_path), dtype=torch.long)

    # Build features in the exact test order
    def fix_dssp_shape(d):
        d = np.array(d)
        if d.ndim == 1: d = d[:, None]
        if d.shape[1] < 9:
            d = np.pad(d, ((0, 0), (0, 9 - d.shape[1])), mode="constant")
        elif d.shape[1] > 9:
            d = d[:, :9]
        return d

    seq_features, pssm_features, dssp_features, go_pair_features, struct_features, label_list = [], [], [], [], [], []
    for _, row in test_pairs.iterrows():
        i, j = row['Protein_A'], row['Protein_B']
        label_list.append(row['label'])

        g_i, g_j = structure_graphs[i], structure_graphs[j]
        struct_features.append([g_i, g_j])

        idx_i, idx_j = ids.index(i), ids.index(j)
        seq_i, seq_j   = sequence_data[idx_i], sequence_data[idx_j]
        pssm_i, pssm_j = pssm_data[i], pssm_data[j]
        dssp_i, dssp_j = fix_dssp_shape(dssp_data[i]), fix_dssp_shape(dssp_data[j])

        go_i, go_j = go_features[idx_i], go_features[idx_j]
        go_concat = np.concatenate([go_i, go_j])

        seq_features.append(np.concatenate([seq_i, seq_j], axis=0))
        pssm_features.append(np.concatenate([pssm_i, pssm_j], axis=0))
        dssp_features.append(np.concatenate([dssp_i, dssp_j], axis=0))
        go_pair_features.append(go_concat)

    max_len = min(1000, max([x.shape[0] for x in seq_features]))

    # Collate closure so we can inject go_edge_index cleanly
    def make_collate_fn(go_edge_index):
        def _collate(batch):
            xs, ys = zip(*batch)
            out = {}
            for k in xs[0]:
                if k == "struct":
                    graphs = list(itertools.chain.from_iterable([x[k] for x in xs]))
                    out[k] = Batch.from_data_list(graphs)
                else:
                    out[k] = torch.stack([x[k] for x in xs])
            out["go_edge_index"] = go_edge_index
            out["go_batch"] = torch.arange(out["go_feat"].shape[0])
            return out, torch.stack(ys).float()
        return _collate

    test_set = PPIPairDataset(
        seq_features, pssm_features, dssp_features, go_pair_features, struct_features, label_list, max_len)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                             collate_fn=make_collate_fn(go_edge_index_tensor))

    # Build model config identical to training
    example_struct = test_set[0][0]["struct"]
    x_s_dim = example_struct[0].x_s.shape[1]
    x_v_dim = example_struct[0].x_v.shape[1] if example_struct[0].x_v.ndim == 3 else 1
    edge_s_dim = example_struct[0].edge_attr[0].shape[1]
    edge_v_dim = example_struct[0].edge_attr[1].shape[1] if example_struct[0].edge_attr[1].ndim > 1 else 0

    config = {
        'seq_input_dim': test_set[0][0]["seq_feat"].shape[1],
        'go_input_dim':  test_set[0][0]["go_feat"].shape[0],
        'num_go_terms':  test_set[0][0]["go_feat"].shape[0],
        'mode': 'full',
        'num_labels': 1,
        'node_dims': (x_s_dim, x_v_dim),
        'edge_dims': (edge_s_dim, edge_v_dim),
    }

    from model import PertiNet
    model = PertiNet(config).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Predict
    all_logits, all_labels, all_disturb = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            sb = x["struct"]
            x["x_s"] = sb.x_s
            x["x_v"] = sb.x_v
            x["edge_index"] = sb.edge_index
            x["edge_attr"] = sb.edge_attr   # tuple
            x["batch"] = sb.batch
            del x["struct"]

            for k in x:
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].to(device)
            y = y.to(device)

            logits, disturb_score, *_ = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_disturb.append(disturb_score.cpu().numpy())

    y_logit = np.concatenate(all_logits, axis=0).ravel()
    y_prob  = expit(y_logit)  # convert logits → probabilities in [0,1]
    y_true  = np.concatenate(all_labels, axis=0).ravel()

    # Threshold search on probabilities
    best = dict(f1=0.0, mcc=0.0, thr=0.5, acc=0.0, prec=0.0, rec=0.0, auprc=0.0)
    for thr in np.linspace(0.0, 1.0, 101):
        y_pred = (y_prob > thr).astype(int)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
        acc = accuracy_score(y_true, y_pred)
        prec= precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        auprc = average_precision_score(y_true, y_prob)
        if f1 > best["f1"]:
            best.update(dict(f1=f1, mcc=mcc, thr=thr, acc=acc, prec=prec, rec=rec, auprc=auprc))

    print(f"[Test] ACC={best['acc']:.4f} P={best['prec']:.4f} R={best['rec']:.4f} "
          f"F1={best['f1']:.4f} AUPRC={best['auprc']:.4f} MCC={best['mcc']:.4f} (thr={best['thr']:.2f})")

    # Save metrics text
    txt_path = os.path.join(model_dir, "test_pred_results109.txt")
    with open(txt_path, "w") as f:
        f.write(f"ACC: {best['acc']:.4f}\nPrecision: {best['prec']:.4f}\nRecall: {best['rec']:.4f}\n")
        f.write(f"F1: {best['f1']:.4f}\nAUPRC: {best['auprc']:.4f}\nMCC: {best['mcc']:.4f}\n")
        f.write(f"Best_thr(prob): {best['thr']:.2f}\n")
    print(f"Saved metrics → {txt_path}")

    # Save per-pair table aligned with test_pairs
    disturb_scores = np.concatenate(all_disturb, axis=0).ravel()
    out_df = test_pairs.copy()
    out_df["pred_score"] = y_prob          # probability (not raw logit)
    out_df["disturb_score"] = disturb_scores
    out_df["label"] = y_true
    out_csv = os.path.join(model_dir, "test_pairs_with_disturb_scores.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved per-pair scores → {out_csv}")

    # Histograms
    out_dir = os.path.join(model_dir, "score_hist"); os.makedirs(out_dir, exist_ok=True)
    plt.figure(); plt.hist(y_logit, bins=50); plt.title('logit'); plt.xlabel('logit'); plt.ylabel('Count')
    plt.savefig(os.path.join(out_dir, "logit_hist.png")); plt.close()
    plt.figure(); plt.hist(y_prob, bins=50); plt.title('sigmoid(logit)'); plt.xlabel('prob'); plt.ylabel('Count')
    plt.savefig(os.path.join(out_dir, "sigmoid_hist.png")); plt.close()
    print(f"Saved histograms → {out_dir}")


if __name__ == "__main__":
    predict()
