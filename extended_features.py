"""
extended_features.py
=======================
Drift-Sense hackathon -- wraps candidate_features.py's standard 10-feature
extraction, adding scale_score_std as an 11th feature
Computed only for the top-K candidates by NCC score (default K=50,
matching the coverage validated earlier: K=50 reaches 72.4% of genuine
ranking-failure cases at 69.1% GT-win rate). Candidates beyond K get a
neutral fallback value (the mean of whatever WAS computed for this pair).

Usage (as a library):
    from extended_features import extract_extended_features
    candidates, ncc, (th, tw) = extract_extended_features(ref, search, top_k=50)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from matching import fft_ncc
from candidate_features import extract_candidate_features, FEATURE_NAMES

SCALES = [9.5, 9.75, 10.0, 10.25, 10.5]
DEFAULT_TOP_K = 50

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + ["scale_score_std"]


def compute_ncc_at_scale(ref, search, zoom_ratio):
    template = zoom(ref, 1.0 / zoom_ratio, order=1)
    th, tw = template.shape
    if th < 2 or tw < 2 or th >= search.shape[0] or tw >= search.shape[1]:
        return None, None, None
    ncc = fft_ncc(search, template)
    return ncc, th, tw


def score_at_physical_location(ncc, th, tw, px, py):
    row = int(round(py - th / 2.0))
    col = int(round(px - tw / 2.0))
    if 0 <= row < ncc.shape[0] and 0 <= col < ncc.shape[1]:
        return float(ncc[row, col])
    return None


def extract_extended_features(ref: np.ndarray, search: np.ndarray, top_k: int = DEFAULT_TOP_K):
    candidates, ncc_10x, (th, tw) = extract_candidate_features(ref, search, return_ncc=True)

    scale_surfaces = {}
    for scale in SCALES:
        if scale == 10.0:
            scale_surfaces[scale] = (ncc_10x, th, tw)
        else:
            scale_surfaces[scale] = compute_ncc_at_scale(ref, search, scale)

    computed_stds = []
    for i, c in enumerate(candidates):
        if i >= top_k:
            break
        px, py = c["col"] + tw / 2.0, c["row"] + th / 2.0
        vals = []
        for scale in SCALES:
            surf, sth, stw = scale_surfaces[scale]
            if surf is None:
                continue
            v = score_at_physical_location(surf, sth, stw, px, py)
            if v is not None:
                vals.append(v)
        std = float(np.std(vals)) if len(vals) >= 2 else None
        computed_stds.append(std)

    valid_stds = [s for s in computed_stds if s is not None]
    fallback = float(np.mean(valid_stds)) if valid_stds else 0.0

    for i, c in enumerate(candidates):
        if i < len(computed_stds) and computed_stds[i] is not None:
            std_val = computed_stds[i]
        else:
            std_val = fallback
        c["features"] = np.append(c["features"], std_val).astype(np.float32)

    return candidates, ncc_10x, (th, tw)
