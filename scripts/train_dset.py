"""Train and evaluate PertiNet-S on the complete prepared Dset collection."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import PertiNetS
from model.dset import load_prepared_record
from model.evaluation import select_mcc_threshold


DEFAULT_PREPARED_ROOT = REPO_ROOT / "data" / "Dset_186_72_PDB164" / "prepared"
FULL_RUN_CONFIG = {
    "data_root": DEFAULT_PREPARED_ROOT,
    "output": Path("results/dset/full_dset_results.json"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 100,
    "patience": 15,
    "accumulate": 4,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "negative_ratio": 1.0,
    "label_smoothing": 0.02,
    "dice_weight": 0.0,
    "ema_decay": 0.995,
    "amp": True,
    "seed": 42,
}
TARGET_STANDARD = {
    "acc": 0.763,
    "precision": 0.413,
    "recall": 0.633,
    "f1": 0.512,
    "auprc": 0.523,
    "mcc": 0.361,
}


def read_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [record_id for record_id in ids if record_id]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight(root, split_ids):
    expected = set().union(*map(set, split_ids.values()))
    overlaps = {
        "train_validation": len(set(split_ids["train"]) & set(split_ids["validation"])),
        "train_test": len(set(split_ids["train"]) & set(split_ids["test"])),
        "validation_test": len(set(split_ids["validation"]) & set(split_ids["test"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"split identifiers overlap: {overlaps}")
    missing = {}
    for folder, suffix in (
        ("seq", ".fasta"),
        ("pssm", ".npy"),
        ("dssp", ".npy"),
        ("labels", ".npy"),
        ("coords", ".npz"),
    ):
        absent = sorted(record_id for record_id in expected if not (root / folder / f"{record_id}{suffix}").is_file())
        if absent:
            missing[folder] = absent
    if missing:
        summary = {folder: len(ids) for folder, ids in missing.items()}
        raise FileNotFoundError(
            f"complete Dset training requires every real feature file; missing counts={summary}. "
            "Run scripts/fetch_dset_coords.py when only coords are missing."
        )
    return {"records": len(expected), "split_overlap": overlaps}


def to_device(record, device):
    graph = record["graph"].to(device)
    return {
        "seq_feat": record["seq_feat"].to(device),
        "seq_mask": record["seq_mask"].to(device),
        "x_s": graph.x_s,
        "x_v": graph.x_v,
        "edge_index": graph.edge_index,
        "edge_attr": graph.edge_attr,
    }, record["labels"].to(device)


def collect_labels(root, ids):
    positives = total = 0
    for record_id in ids:
        labels = np.load(root / "labels" / f"{record_id}.npy", allow_pickle=False)
        positives += int(np.asarray(labels).sum())
        total += int(np.asarray(labels).size)
    return positives, total


def balanced_site_loss(logits, labels, negative_ratio=1.0, label_smoothing=0.0, dice_weight=0.0):
    """Per-protein balanced residue loss; evaluation still uses every residue."""
    labels = labels.float()
    positive = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    negative = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
    if positive.numel() and negative.numel() and negative_ratio > 0:
        keep_negative = min(negative.numel(), max(1, int(round(positive.numel() * negative_ratio))))
        choice = torch.randperm(negative.numel(), device=negative.device)[:keep_negative]
        selected = torch.cat([positive, negative[choice]])
        logits = logits[selected]
        labels = labels[selected]
    smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, smooth_labels)
    if dice_weight <= 0:
        return bce
    probability = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (probability * labels).sum() + 1.0) / (
        probability.sum() + labels.sum() + 1.0
    )
    return bce + float(dice_weight) * dice


class ExponentialMovingAverage:
    """EMA weights for more stable validation and held-out predictions."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def apply(self, model):
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)
        self.backup = None


def evaluate(model, root, ids, device, threshold=None):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for record_id in ids:
            record = load_prepared_record(root, record_id)
            model_input, target = to_device(record, device)
            scores.append(torch.sigmoid(model(model_input)).cpu().numpy())
            labels.append(target.cpu().numpy())
    score = np.concatenate(scores)
    label = np.concatenate(labels).astype(int)
    selected_threshold = select_mcc_threshold(score, label) if threshold is None else float(threshold)
    prediction = (score >= selected_threshold).astype(int)
    metrics = {
        "acc": accuracy_score(label, prediction),
        "precision": precision_score(label, prediction, zero_division=0),
        "recall": recall_score(label, prediction, zero_division=0),
        "f1": f1_score(label, prediction, zero_division=0),
        "auprc": average_precision_score(label, score),
        "mcc": matthews_corrcoef(label, prediction),
        "threshold": selected_threshold,
        "residues": int(label.size),
        "positive_residues": int(label.sum()),
    }
    return {key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in metrics.items()}


def train(args):
    seed_everything(args.seed)
    root = args.data_root
    split_ids = {
        "train": read_ids(root / "train.txt"),
        "validation": read_ids(root / "val.txt"),
        "test": read_ids(root / "test.txt"),
    }
    preflight_report = preflight(root, split_ids)
    device = torch.device(args.device)
    model = PertiNetS({
        "seq_input_dim": 40,
        "node_dims": (9, 1),
        "edge_dims": (1, 1),
    }).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ExponentialMovingAverage(model, args.ema_decay)

    best_state = None
    best_epoch = 0
    best_val_auprc = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_ids = list(split_ids["train"])
        random.shuffle(epoch_ids)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, record_id in enumerate(epoch_ids, 1):
            record = load_prepared_record(root, record_id)
            model_input, target = to_device(record, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(model_input)
                loss = balanced_site_loss(
                    logits,
                    target,
                    negative_ratio=args.negative_ratio,
                    label_smoothing=args.label_smoothing,
                    dice_weight=args.dice_weight,
                ) / args.accumulate
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.accumulate
            if step % args.accumulate == 0 or step == len(epoch_ids):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
        scheduler.step()
        ema.apply(model)
        validation = evaluate(model, root, split_ids["validation"], device)
        ema.restore(model)
        item = {
            "epoch": epoch,
            "train_loss": running_loss / len(epoch_ids),
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
        }
        history.append(item)
        print(json.dumps(item), flush=True)
        if validation["auprc"] > best_val_auprc:
            best_val_auprc = validation["auprc"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    validation = evaluate(model, root, split_ids["validation"], device)
    test = evaluate(model, root, split_ids["test"], device, threshold=validation["threshold"])
    differences = {key: test[key] - value for key, value in TARGET_STANDARD.items()}
    result = {
        "dataset": "Dset_186_72_PDB164",
        "data_root": str(root),
        "seed": args.seed,
        "device": str(device),
        "training_controls": {
            "negative_ratio": args.negative_ratio,
            "label_smoothing": args.label_smoothing,
            "dice_weight": args.dice_weight,
            "ema_decay": args.ema_decay,
            "cosine_learning_rate": True,
            "mixed_precision": use_amp,
            "gradient_clip": args.grad_clip,
        },
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "split_protocol": "DeepPPISP 352-development/70-test with fixed-seed 50-protein validation holdout",
        "split_protocol_warning": (
            None
            if [len(split_ids[key]) for key in ("train", "validation", "test")] == [302, 50, 70]
            else "Expected the documented DeepPPISP-referenced 302/50/70 protein-level split."
        ),
        "preflight": preflight_report,
        "best_epoch": best_epoch,
        "validation": validation,
        "test": test,
        "manuscript_standard_target": TARGET_STANDARD,
        "test_minus_manuscript": differences,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model_state": best_state, "result": result}, args.output.with_suffix(".pt"))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=FULL_RUN_CONFIG["data_root"])
    parser.add_argument("--output", type=Path, default=FULL_RUN_CONFIG["output"])
    parser.add_argument("--device", default=FULL_RUN_CONFIG["device"])
    parser.add_argument("--epochs", type=int, default=FULL_RUN_CONFIG["epochs"])
    parser.add_argument("--patience", type=int, default=FULL_RUN_CONFIG["patience"])
    parser.add_argument("--accumulate", type=int, default=FULL_RUN_CONFIG["accumulate"])
    parser.add_argument("--lr", type=float, default=FULL_RUN_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=FULL_RUN_CONFIG["weight_decay"])
    parser.add_argument("--grad-clip", type=float, default=FULL_RUN_CONFIG["grad_clip"])
    parser.add_argument("--negative-ratio", type=float, default=FULL_RUN_CONFIG["negative_ratio"])
    parser.add_argument("--label-smoothing", type=float, default=FULL_RUN_CONFIG["label_smoothing"])
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=FULL_RUN_CONFIG["dice_weight"],
        help="Optional Dice auxiliary weight; keep 0 for the manuscript-aligned BCE run.",
    )
    parser.add_argument("--ema-decay", type=float, default=FULL_RUN_CONFIG["ema_decay"])
    parser.set_defaults(amp=FULL_RUN_CONFIG["amp"])
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=FULL_RUN_CONFIG["seed"])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
