# Dset executable example

This directory contains three real residue-level records, one from each source
subset of the manuscript's merged Dset_186_72_PDB164 collection: Dset_72,
PDBset_164 and Dset_186. Each record includes FASTA sequence, PSSM, DSSP,
binary interface labels and real C-alpha coordinates.

The example verifies data loading, 40-channel sequence construction,
10-Angstrom C-alpha graph construction, and PertiNet-S forward/backward
execution. It is not a training partition and must not be used to reproduce or
claim the full benchmark metrics.

Run from the repository root:

```bash
python examples/dset_smoke.py
```
