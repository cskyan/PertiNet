# RBP400 retained experiment

This workflow matches the case-study scope in the manuscript. It is separate
from the tri-modal Bernett benchmark and from the matched RBP400 ablations.

1. Build case-excluded RBP400-Expanded splits:

```bash
python -m experiments.rbp400.prepare_case_safe_pairs \
  --expanded-root /path/to/RBP400_Expanded \
  --rbp400-root /path/to/RBP400 \
  --cluster-id30 /path/to/clusters_id30.tsv \
  --cluster-id20 /path/to/clusters_id20.tsv \
  --output-root outputs/rbp400/prepared
```

2. Fit the five frozen ESM/PCA-logistic development models:

```bash
python -m experiments.rbp400.train_esm_case_model \
  --split-root outputs/rbp400/prepared \
  --esm-cache /path/to/esm_cache \
  --output-root outputs/rbp400/models \
  --device cuda:0
```

3. Score the two predefined RBP400 candidate universes:

```bash
python -m experiments.rbp400.predict_hcc_lung_linear \
  --rbp400-root /path/to/RBP400 \
  --esm-cache /path/to/esm_cache \
  --model-root outputs/rbp400/models \
  --output-root outputs/rbp400/case_study
```

Use `--smoke-pairs 100` in step 3 for a bounded execution check. The locked
summary under `case_study/rbp400/results/` was produced without this limit.
