"""Validation-controlled operating boundary and held-out confidence utilities."""

import numpy as np
from sklearn.metrics import matthews_corrcoef


def select_mcc_threshold(validation_scores, validation_labels, grid_size=1001):
    """Choose a score threshold on validation data only by maximum MCC."""
    scores = np.asarray(validation_scores, dtype=float).reshape(-1)
    labels = np.asarray(validation_labels, dtype=int).reshape(-1)
    if scores.size != labels.size or scores.size == 0:
        raise ValueError("validation scores and labels must have the same nonzero length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("validation labels must be binary")
    candidates = np.linspace(0.0, 1.0, int(grid_size))
    mcc_values = np.array([
        matthews_corrcoef(labels, scores >= threshold)
        for threshold in candidates
    ])
    return float(candidates[int(np.nanargmax(mcc_values))])


def apply_operating_boundary(scores, threshold):
    """Return predictions and distance from the frozen validation threshold."""
    scores = np.asarray(scores, dtype=float)
    threshold = float(threshold)
    return (scores >= threshold).astype(np.int64), np.abs(scores - threshold)
