"""Materialize the manuscript Dset fused source and real C-alpha coordinates."""

import argparse
import json
import pickle
import re
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DSET_ROOT = REPO_ROOT / "data" / "Dset_186_72_PDB164"
DEFAULT_SOURCE = BUNDLED_DSET_ROOT / "source"
DEFAULT_OUTPUT = BUNDLED_DSET_ROOT / "prepared"
AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
DATASET_ORDER = ("dset72", "dset164", "dset186")


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def safe_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def sequence_text(encoded):
    return "".join(AA_ORDER[int(value)] if 0 <= int(value) < len(AA_ORDER) else "X" for value in encoded)


def grouped_source_records(rows):
    groups = []
    start = 0
    previous = None
    for index, row in enumerate(rows):
        key = (row[1], row[3], row[4], row[5])
        if previous is not None and key != previous:
            groups.append((*previous, index - start))
            start = index
        previous = key
    if previous is not None:
        groups.append((*previous, len(rows) - start))
    return groups


def parse_structure_id(raw_id):
    text = str(raw_id).strip()
    if "_" in text:
        pdb_id, chains = text.split("_", 1)
    else:
        pdb_id, chains = text[:4], text[4:]
    return pdb_id.lower(), "".join(char for char in chains if char.isalnum())


def build_manifest(source):
    grouped = grouped_source_records(load_pickle(source / "all_dset_list.pkl"))
    by_dataset = {name: [] for name in DATASET_ORDER}
    for sample_no, dataset, raw_id, declared_length, row_count in grouped:
        if dataset in by_dataset:
            by_dataset[dataset].append((sample_no, raw_id, declared_length, row_count))
    manifest = []
    fused_index = 0
    for dataset in DATASET_ORDER:
        for local_index, (_, raw_id, declared_length, row_count) in enumerate(by_dataset[dataset]):
            pdb_id, chains = parse_structure_id(raw_id)
            manifest.append({
                "fused_idx": fused_index,
                "pid": safe_id(f"dest{fused_index:04d}_{dataset}_{raw_id}"),
                "dataset": dataset,
                "local_idx": local_index,
                "raw_id": str(raw_id),
                "pdb_id": pdb_id,
                "chains": chains,
                "declared_len": int(declared_length),
                "group_rows": int(row_count),
            })
            fused_index += 1
    return manifest


def write_fasta(path, record_id, sequence):
    lines = [f">{record_id}"] + [sequence[index:index + 80] for index in range(0, len(sequence), 80)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize(source, output, manifest):
    for folder in ("seq", "pssm", "dssp", "labels", "coords", "pdb_raw"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    fused = {
        "sequence": load_pickle(source / "fused_sequence_data.pkl"),
        "pssm": load_pickle(source / "fused_pssm_data.pkl"),
        "dssp": load_pickle(source / "fused_dssp_data.pkl"),
        "labels": load_pickle(source / "fused_label.pkl"),
    }
    if any(len(value) != len(manifest) for value in fused.values()):
        raise ValueError("fused feature collections and manifest do not have the same record count")
    for row in manifest:
        index, record_id = row["fused_idx"], row["pid"]
        write_fasta(output / "seq" / f"{record_id}.fasta", record_id, sequence_text(fused["sequence"][index]))
        np.save(output / "pssm" / f"{record_id}.npy", np.asarray(fused["pssm"][index], dtype=np.float32))
        np.save(output / "dssp" / f"{record_id}.npy", np.asarray(fused["dssp"][index], dtype=np.float32))
        np.save(output / "labels" / f"{record_id}.npy", np.asarray(fused["labels"][index], dtype=np.float32))
    split_files = {
        "train.txt": "fused_training_list.pkl",
        "val.txt": "fused_validing_list.pkl",
        "test.txt": "fused_test_list.pkl",
    }
    for destination, source_name in split_files.items():
        indices = [int(value) for value in load_pickle(source / source_name)]
        (output / destination).write_text(
            "".join(f"{manifest[index]['pid']}\n" for index in indices), encoding="utf-8"
        )
    (output / "all_ids.txt").write_text(
        "".join(f"{row['pid']}\n" for row in manifest), encoding="utf-8"
    )


def download_pdb(pdb_id, destination, timeout, retries):
    if destination.is_file() and destination.stat().st_size:
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PertiNet-release/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()
            if content:
                destination.write_bytes(content)
                return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.0 + attempt)


def extract_chain_ca(path, chains):
    coordinates, seen = [], set()
    wanted = set(chains)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21:22].strip()
        if wanted and chain not in wanted:
            continue
        residue = (chain, line[22:27])
        if residue in seen:
            continue
        seen.add(residue)
        coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(coordinates, dtype=np.float32).reshape(-1, 3)


def materialize_coordinates(output, manifest, timeout, retries, bundled_coordinates=None):
    for number, row in enumerate(manifest, 1):
        coordinate_path = output / "coords" / f"{row['pid']}.npz"
        if coordinate_path.is_file() and coordinate_path.stat().st_size:
            row["coord_status"] = "ok"
            continue
        bundled_path = (
            Path(bundled_coordinates) / f"{row['pid']}.npz"
            if bundled_coordinates is not None else None
        )
        if bundled_path is not None and bundled_path.is_file():
            shutil.copy2(str(bundled_path), str(coordinate_path))
            row["coord_status"] = "ok"
            continue
        pdb_path = output / "pdb_raw" / f"{row['pdb_id']}.pdb"
        try:
            download_pdb(row["pdb_id"], pdb_path, timeout, retries)
            coordinates = extract_chain_ca(pdb_path, row["chains"])
            if not len(coordinates):
                raise ValueError("no requested-chain C-alpha atoms")
            np.savez_compressed(coordinate_path, coords=coordinates)
            row["coord_status"] = "ok"
        except Exception as error:
            row["coord_status"] = f"error: {error}"
        if number % 25 == 0:
            print(f"[coords] {number}/{len(manifest)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.set_defaults(download_coordinates=True)
    parser.add_argument(
        "--download-coordinates",
        dest="download_coordinates",
        action="store_true",
    )
    parser.add_argument(
        "--no-download-coordinates",
        dest="download_coordinates",
        action="store_false",
    )
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    manifest = build_manifest(args.source_root)
    materialize(args.source_root, args.output_root, manifest)
    if args.download_coordinates:
        materialize_coordinates(
            args.output_root,
            manifest,
            args.timeout,
            args.retries,
            bundled_coordinates=args.source_root / "coords",
        )
    (args.output_root / "dest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    success = sum(row.get("coord_status") == "ok" for row in manifest)
    print(json.dumps({"records": len(manifest), "coordinates_ready": success, "status": "PASS" if success == len(manifest) else "INCOMPLETE"}, indent=2))


if __name__ == "__main__":
    main()
