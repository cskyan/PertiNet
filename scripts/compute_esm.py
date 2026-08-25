#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute ESM-2 per-residue embeddings as one .npy file per protein."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch


DEFAULT_ESM2_650M_PT = "esm2_t33_650M_UR50D.pt"


def read_fasta(path: Path):
    seqs = {}
    cur = None
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(re.sub("[^A-Za-z]", "", line).upper())
        if cur is not None:
            seqs[cur] = "".join(buf)
    return seqs


def prepare_local_esm_checkpoint(local_model: str):
    src = Path(local_model)
    if not src.exists():
        print(f"[esm][warn] local model not found, fair-esm may download: {src}", flush=True)
        return
    cache_dir = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))) / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / "esm2_t33_650M_UR50D.pt"
    if dst.exists():
        try:
            if dst.resolve() == src.resolve():
                print(f"[esm] checkpoint cache already points to local model: {dst}", flush=True)
                return
        except Exception:
            pass
        if dst.stat().st_size == src.stat().st_size:
            print(f"[esm] checkpoint cache already exists: {dst}", flush=True)
            return
        backup = dst.with_suffix(dst.suffix + ".partial_or_old")
        dst.rename(backup)
        print(f"[esm] moved old checkpoint to {backup}", flush=True)
    try:
        dst.symlink_to(src)
        print(f"[esm] linked local checkpoint: {dst} -> {src}", flush=True)
    except Exception as exc:
        import shutil

        print(f"[esm][warn] symlink failed ({exc}); copying checkpoint instead.", flush=True)
        shutil.copy2(src, dst)
        print(f"[esm] copied local checkpoint: {dst}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Precompute ESM-2 650M residue embeddings.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--repr-layer", type=int, default=33)
    p.add_argument("--local-model", default=DEFAULT_ESM2_650M_PT)
    args = p.parse_args()

    prepare_local_esm_checkpoint(args.local_model)

    try:
        import esm
    except ImportError as exc:
        raise SystemExit("Missing dependency: install fair-esm in this environment first.") from exc

    fasta = Path(args.fasta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    seqs = read_fasta(fasta)
    uids = [uid for uid in seqs if not (out_dir / f"{uid}.npy").exists()]
    print(f"[esm] fasta={fasta} total={len(seqs)} to_compute={len(uids)} out={out_dir}", flush=True)

    for i in range(0, len(uids), int(args.batch)):
        chunk = uids[i : i + int(args.batch)]
        data = [(uid, seqs[uid][: int(args.max_len)]) for uid in chunk]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            rep = model(tokens, repr_layers=[int(args.repr_layer)])["representations"][int(args.repr_layer)]
        for k, uid in enumerate(chunk):
            L = min(len(seqs[uid]), int(args.max_len))
            emb = rep[k, 1 : L + 1].detach().cpu().float().numpy().astype(np.float32)
            np.save(out_dir / f"{uid}.npy", emb)
        print(f"[esm] {min(i + len(chunk), len(uids))}/{len(uids)} done", flush=True)


if __name__ == "__main__":
    main()
