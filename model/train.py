# train.py
import os
import itertools
import pickle
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, average_precision_score,
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch

from model import PertiNet


# ===== Paths (EDIT ONLY THESE TWO) =====
base_dir = "..."   # was: /srv/storage/.../RBP109_raw_features
model_dir = "..."  # was: /srv/storage/.../model
os.makedirs(model_dir, exist_ok=True)

model_save_path = os.path.join(model_dir, "fused109.best.pth")
sequence_path   = os.path.join(base_dir, "sequence_onehot.npy")
pssm_path       = os.path.join(base_dir, "pssm_109.npz")
dssp_path       = os.path.join(base_dir, "dssp_109.npz")
label_path      = os.path.join(base_dir, "ppi_labels_balanced.npy")
go_edge_npy     = os.path.join(base_dir, "go_term_edge_index.npy")
struct_pkl      = os.path.join(base_dir, "protein_graphs_109.pkl")


# ===== Utils =====
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_2d(a: np.ndarray, dim: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[1] < dim:
        a = np.concatenate([a, np.zeros((a.shape[0], dim - a.shape[1]), a.dtype)], axis=1)
    elif a.shape[1] > dim:
        a = a[:, :dim]
    return a


def to_gvp_vec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:, None, :]
    return np.zeros((arr.shape[0], 1, 3), dtype=np.float32)


def make_edge_attr(edge_scalar: np.ndarray, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    es = ensure_2d(edge_scalar, target_dim)
    num_e = es.shape[0]
    ev = np.zeros((num_e, 0, 3), dtype=np.float32)
    return torch.tensor(es, dtype=torch.float32), torch.tensor(ev, dtype=torch.float32)


# ===== Dataset =====
class PPIPairDataset(Dataset):
    def __init__(self, seq_list, pssm_list, dssp_list, go_list, struct_list, labels, max_len=500):
        self.seq_list, self.pssm_list, self.dssp_list = seq_list, pssm_list, dssp_list
        self.go_list, self.struct_list, self.labels = go_list, struct_list, labels
        self.max_len = max_len
        assert len(self.labels) == len(self.struct_list)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        seq = np.asarray(self.seq_list[idx])
        pssm = np.asarray(self.pssm_list[idx])
        dssp = np.asarray(self.dssp_list[idx])
        go   = np.asarray(self.go_list[idx])
        g_i, g_j = self.struct_list[idx]
        label = float(self.labels[idx])

        x_s_dim = max(np.asarray(g_i["x_s"]).shape[1], np.asarray(g_j["x_s"]).shape[1])
        e_dim   = max(np.asarray(g_i["edge_attr"]).shape[1], np.asarray(g_j["edge_attr"]).shape[1])
        es_i, ev_i = make_edge_attr(g_i["edge_attr"], e_dim)
        es_j, ev_j = make_edge_attr(g_j["edge_attr"], e_dim)

        data_i = Data(
            x_s=torch.tensor(ensure_2d(g_i["x_s"], x_s_dim), dtype=torch.float32),
            x_v=torch.tensor(to_gvp_vec(g_i["x_v"]), dtype=torch.float32),
            edge_index=torch.tensor(g_i["edge_index"], dtype=torch.long),
            edge_attr=(es_i, ev_i),
        )
        data_j = Data(
            x_s=torch.tensor(ensure_2d(g_j["x_s"], x_s_dim), dtype=torch.float32),
            x_v=torch.tensor(to_gvp_vec(g_j["x_v"]), dtype=torch.float32),
            edge_index=torch.tensor(g_j["edge_index"], dtype=torch.long),
            edge_attr=(es_j, ev_j),
        )

        # concat along length; crop/pad to max_len
        min_len = min(len(seq), len(pssm), len(dssp))
        seq, pssm, dssp = seq[:min_len], pssm[:min_len], dssp[:min_len]
        x_raw = np.concatenate([seq, pssm, dssp], axis=-1)
        x_raw = x_raw[:self.max_len]
        if x_raw.shape[0] < self.max_len:
            pad = self.max_len - x_raw.shape[0]
            x_raw = np.pad(x_raw, ((0, pad), (0, 0)))

        return {
            "seq_feat": torch.tensor(x_raw, dtype=torch.float32),
            "go_feat":  torch.tensor(go, dtype=torch.float32),
            "struct":   [data_i, data_j],
        }, torch.tensor([label], dtype=torch.float32)


def collate_fn(batch):
    xs, ys = zip(*batch)
    out = {}
    for k in xs[0]:
        if k == "struct":
            graphs = list(itertools.chain.from_iterable([x[k] for x in xs]))
            out[k] = Batch.from_data_list(graphs)
        else:
            out[k] = torch.stack([x[k] for x in xs])
    return out, torch.stack(ys)


# ===== Auxiliary loss =====
class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5, embed_dim=64):
        super().__init__()
        self.margin1, self.margin2 = margin1, margin2
        self.p_anchor   = nn.Linear(64, embed_dim)
        self.p_positive = nn.Linear(128, embed_dim)  # struct is 128 after pair-concat
        self.p_negative = nn.Linear(64, embed_dim)

    def forward(self, anchor, positive, negative, negative2):
        a = self.p_anchor(anchor)
        p = self.p_positive(positive)
        n = self.p_negative(negative)
        n2 = self.p_negative(negative2)
        d_ap = F.pairwise_distance(a, p)
        d_an = F.pairwise_distance(a, n)
        d_nn = F.pairwise_distance(n, n2)
        return (F.relu(d_ap - d_an + self.margin1) +
                F.relu(d_ap - d_nn + self.margin2)).mean()


# ===== Train =====
def train():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(base_dir, "human_reviewed_uniprot_ids.txt")) as f:
        ids = [x.strip() for x in f if x.strip()]

    # load features
    seq_np   = np.load(sequence_path, allow_pickle=True)
    pssm_np  = np.load(pssm_path, allow_pickle=True)
    dssp_np  = np.load(dssp_path, allow_pickle=True)
    go_mhot  = np.load(os.path.join(base_dir, "go_multi_hot_109.npy"))
    go_edge_index = torch.tensor(np.load(go_edge_npy), dtype=torch.long)

    with open(struct_pkl, "rb") as f:
        struct_dict = pickle.load(f)

    df = pd.read_csv(label_path.replace(".npy", ".csv"))

    # build pair-wise features
    seq_list, pssm_list, dssp_list, go_list, struct_list, y_list = [], [], [], [], [], []
    for _, row in df.iterrows():
        i, j = row['Protein_A'], row['Protein_B']
        y_list.append(int(row['label']))
        struct_list.append([struct_dict[i], struct_dict[j]])

        idx_i, idx_j = ids.index(i), ids.index(j)
        seq_i, seq_j = seq_np[idx_i], seq_np[idx_j]
        pssm_i = pssm_np.get(i, np.zeros((len(seq_i), 20), dtype=np.float32))
        pssm_j = pssm_np.get(j, np.zeros((len(seq_j), 20), dtype=np.float32))
        dssp_i = dssp_np.get(i, np.zeros((len(seq_i), 9), dtype=np.float32))
        dssp_j = dssp_np.get(j, np.zeros((len(seq_j), 9), dtype=np.float32))
        # ensure dims
        seq_i, seq_j = ensure_2d(seq_i, 20), ensure_2d(seq_j, 20)
        pssm_i, pssm_j = ensure_2d(pssm_i, 20), ensure_2d(pssm_j, 20)
        dssp_i, dssp_j = ensure_2d(dssp_i, 9),  ensure_2d(dssp_j, 9)
        go_i, go_j = go_mhot[idx_i], go_mhot[idx_j]

        seq_list.append(np.concatenate([seq_i,  seq_j], axis=0))
        pssm_list.append(np.concatenate([pssm_i, pssm_j], axis=0))
        dssp_list.append(np.concatenate([dssp_i, dssp_j], axis=0))
        go_list.append(np.concatenate([go_i, go_j], axis=0))

    # split by unique pair
    df['pair'] = df.apply(lambda r: '-'.join(sorted([r['Protein_A'], r['Protein_B']])), axis=1)
    pairs = df['pair'].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(pairs)
    n = len(pairs)
    train_pairs = set(pairs[: int(0.8 * n)])
    val_pairs   = set(pairs[int(0.8 * n): int(0.9 * n)])
    test_pairs  = set(pairs[int(0.9 * n):])

    tr_idx = df[df['pair'].isin(train_pairs)].index.tolist()
    va_idx = df[df['pair'].isin(val_pairs)].index.tolist()
    te_idx = df[df['pair'].isin(test_pairs)].index.tolist()

    np.save(os.path.join(model_dir, "train_idx109.npy"), tr_idx)
    np.save(os.path.join(model_dir, "val_idx109.npy"), va_idx)
    np.save(os.path.join(model_dir, "test_idx109.npy"), te_idx)

    max_len = min(1000, max(x.shape[0] for x in seq_list))

    def build(idxs):
        return PPIPairDataset(
            [seq_list[i] for i in idxs],
            [pssm_list[i] for i in idxs],
            [dssp_list[i] for i in idxs],
            [go_list[i]  for i in idxs],
            [struct_list[i] for i in idxs],
            [y_list[i] for i in idxs],
            max_len=max_len
        )

    ds_tr, ds_va, ds_te = build(tr_idx), build(va_idx), build(te_idx)

    # loaders
    weights = torch.ones(len(tr_idx), dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    ld_tr = DataLoader(ds_tr, batch_size=32, sampler=sampler, collate_fn=collate_fn)
    ld_va = DataLoader(ds_va, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # infer dims from one sample
    ex_struct = ds_tr[0][0]["struct"]
    x_s_dim = ex_struct[0].x_s.size(1)
    x_v_dim = ex_struct[0].x_v.size(1) if ex_struct[0].x_v.dim() == 3 else 1
    e_s_dim = ex_struct[0].edge_attr[0].size(1)
    e_v_dim = ex_struct[0].edge_attr[1].size(1) if ex_struct[0].edge_attr[1].dim() > 1 else 0

    cfg = dict(
        seq_input_dim=ds_tr[0][0]["seq_feat"].size(1),
        go_input_dim=go_mhot.shape[1],
        num_go_terms=go_mhot.shape[1],
        mode="full",
        num_labels=1,
        node_dims=(x_s_dim, x_v_dim),
        edge_dims=(e_s_dim, e_v_dim),
    )
    model = PertiNet(cfg).to(device)

    # loss/opt
    y_tr = np.array([y_list[i] for i in tr_idx])
    pos, neg = y_tr.sum(), y_tr.size - y_tr.sum()
    pos_w = torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    quad = QuadrupletLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)

    best_acc, patience, bad = 0.0, 5, 0
    EPOCHS, L_DIST, L_QUAD = 30, 0.2, 0.1

    for ep in range(1, EPOCHS + 1):
        model.train()
        epoch_disturb = []

        for x, y in ld_tr:
            sb = x["struct"].to(device)
            x_dev = {
                "seq_feat": x["seq_feat"].to(device),
                "go_feat":  x["go_feat"].to(device),
                "x_s": sb.x_s, "x_v": sb.x_v,
                "edge_index": sb.edge_index, "edge_attr": sb.edge_attr,
                "batch": sb.batch,
                "go_edge_index": go_edge_index.to(device),
            }
            y = y.to(device)

            logits, disturb, h, h_seq, h_struct, h_func, _ = model(x_dev)
            epoch_disturb.extend(disturb.detach().cpu().numpy())

            loss_main = bce(logits.view(-1), y.view(-1))
            perm = torch.randperm(h_func.size(0), device=device)
            loss_quad = quad(h_seq, h_struct, h_func, h_func[perm])
            loss_dist = -disturb.mean()
            loss = loss_main + L_DIST * loss_dist + L_QUAD * loss_quad

            opt.zero_grad(); loss.backward(); opt.step()

        # save disturbance scores of this epoch
        np.savetxt(os.path.join(model_dir, f"disturb_scores_epoch{ep}.txt"),
                   np.array(epoch_disturb).ravel(), fmt="%.6f")

        # validation
        model.eval()
        logits_all, labels_all = [], []
        with torch.no_grad():
            for x, y in ld_va:
                sb = x["struct"].to(device)
                x_dev = {
                    "seq_feat": x["seq_feat"].to(device),
                    "go_feat":  x["go_feat"].to(device),
                    "x_s": sb.x_s, "x_v": sb.x_v,
                    "edge_index": sb.edge_index, "edge_attr": sb.edge_attr,
                    "batch": sb.batch,
                    "go_edge_index": go_edge_index.to(device),
                }
                y = y.to(device)
                logits, *_ = model(x_dev)
                logits_all.append(logits.cpu().numpy())
                labels_all.append(y.cpu().numpy())

        y_logit = np.concatenate(logits_all).ravel()
        y_prob  = 1.0 / (1.0 + np.exp(-y_logit))
        y_true  = np.concatenate(labels_all).ravel()

        thr = float(model.learnable_thr.item())
        y_pred = (y_logit > thr).astype(int)  # threshold on logits as designed

        acc = accuracy_score(y_true, y_pred)
        prec= precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred) if np.unique(y_true).size > 1 else 0.0
        auprc = average_precision_score(y_true, y_prob) if np.unique(y_true).size > 1 else 0.0

        print(f"[Epoch {ep}] ACC={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} "
              f"AUPRC={auprc:.4f} MCC={mcc:.4f} (thr={thr:.2f})")

        if acc > best_acc:
            best_acc, bad = acc, 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Saved best → {model_save_path}")
        else:
            bad += 1
            print(f"(no improve {bad}/{patience})")
        sch.step()
        if bad >= patience:
            print("Early stop.")
            break


if __name__ == "__main__":
    train()

    # save split CSVs with fixed names (for predict.py)
    df = pd.read_csv(label_path.replace(".npy", ".csv"))
    tr = np.load(os.path.join(model_dir, "train_idx109.npy"))
    va = np.load(os.path.join(model_dir, "val_idx109.npy"))
    te = np.load(os.path.join(model_dir, "test_idx109.npy"))

    df.iloc[tr][['Protein_A', 'Protein_B', 'label']].to_csv(
        os.path.join(model_dir, "train_pairs109.csv"), index=False)
    df.iloc[va][['Protein_A', 'Protein_B', 'label']].to_csv(
        os.path.join(model_dir, "val_pairs109.csv"), index=False)
    df.iloc[te][['Protein_A', 'Protein_B', 'label']].to_csv(
        os.path.join(model_dir, "test_pairs109.csv"), index=False)

    # quick preview
    print("=== Sample of test pairs ===")
    for k in range(min(3, len(te))):
        row = df.iloc[te[k]]
        print(f"{row['Protein_A']} - {row['Protein_B']} (label={row['label']})")
