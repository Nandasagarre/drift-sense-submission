"""
candidate_features.py
========================
Drift-Sense hackathon -- extract the 10 engineered features used by the ranker. The ranker is a
small MLP that takes these 10 features as input and outputs a single confidence score.


FEATURES (in this exact order -- FEATURE_NAMES is the source of truth):
    ncc_score             -- raw NCC correlation score at this candidate
    rank                  -- position among all candidates, sorted by score
                              (0 = top-scoring candidate)
    score_gap_from_top     -- top_score - this_candidate's score (0 for the
                              top candidate itself)
    curvature_x, curvature_y -- directional second-derivative sharpness of
                              the local NCC surface (see NOTES.md section 4 --
                              this is the one signal independently confirmed
                              on two different codebases)
    symmetry               -- basin symmetry (weaker signal, kept for
                              completeness -- real but modest, per the
                              earlier tie-bias-corrected finding)
    anisotropy              -- |curv_x - curv_y| / (|curv_x| + |curv_y|)
    basin_width             -- radial width of the local correlation basin
    center_distance_norm    -- distance from search-image center, normalized
                              (kept as a SIGNAL for the model to weigh, not a
                              hard rule -- a hard center-distance rule alone
                              was already confirmed to lose to argmax)
    n_candidates_norm       -- log-scaled candidate-pool size (ambiguity level)

Usage (as a library -- not meant to be run standalone):
    from candidate_features import extract_candidate_features, FEATURE_NAMES
    candidates = extract_candidate_features(ref, search)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import laplace

from matching import downscale_reference, fft_ncc, non_max_suppression

ZOOM_RATIO = 10
NMS_MIN_DISTANCE_DIVISOR = 3
NMS_SCORE_THRESHOLD = 0.1
MAX_PEAKS = 2000  # matches the FIXED (not the buggy default-20) value used
                   # throughout the rest of this project.
WINDOW_HALF = 10

FEATURE_NAMES = [
    "ncc_score", "rank", "score_gap_from_top",
    "curvature_x", "curvature_y", "symmetry", "anisotropy",
    "basin_width", "center_distance_norm", "n_candidates_norm",
]


def _extract_window(ncc: np.ndarray, row: int, col: int, half: int = WINDOW_HALF):
    """Returns the window AND the candidate's TRUE local position within
    it. CONFIRMED BUG FIX: near an NCC-surface edge, the window gets
    clipped asymmetrically (e.g. row=3 with half=10 only extends downward,
    giving a 14-row window, not the full 21), so the candidate is NOT at
    window.shape//2 anymore -- verified directly: assumed_cy=7 vs true
    local_cy=3 for a row=3 edge case. Downstream functions must use the
    returned (local_row, local_col), not re-derive it via shape//2."""
    r0, r1 = max(0, row - half), min(ncc.shape[0], row + half + 1)
    c0, c1 = max(0, col - half), min(ncc.shape[1], col + half + 1)
    window = ncc[r0:r1, c0:c1]
    local_row, local_col = row - r0, col - c0
    return window, local_row, local_col


def _directional_curvature(window: np.ndarray, cy: int, cx: int):
    if window.size < 9:
        return 0.0, 0.0
    lap = laplace(window)
    curv_x = curv_y = 0.0
    if 0 < cx < window.shape[1] - 1:
        curv_x = -(window[cy, cx - 1] - 2 * window[cy, cx] + window[cy, cx + 1])
    if 0 < cy < window.shape[0] - 1:
        curv_y = -(window[cy - 1, cx] - 2 * window[cy, cx] + window[cy + 1, cx])
    return float(curv_x), float(curv_y)


def _basin_width_and_symmetry(window: np.ndarray, cy: int, cx: int, drop_frac: float = 0.5):
    peak_val = window[cy, cx]
    min_val = window.min()
    if peak_val <= min_val:
        return 0.0, 0.0
    threshold = peak_val - drop_frac * (peak_val - min_val)

    def walk(dr, dc):
        r, c = cy, cx
        dist = 0
        while True:
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < window.shape[0] and 0 <= c2 < window.shape[1]):
                break
            if window[r2, c2] < threshold:
                break
            r, c = r2, c2
            dist += 1
            if dist > max(window.shape):
                break
        return dist

    w_left, w_right = walk(0, -1), walk(0, 1)
    w_up, w_down = walk(-1, 0), walk(1, 0)
    mean_width = (w_left + w_right + w_up + w_down) / 4.0

    def sym(a, b):
        return 1.0 - abs(a - b) / (a + b) if (a + b) > 0 else 0.0

    symmetry = (sym(w_left, w_right) + sym(w_up, w_down)) / 2.0
    return mean_width, symmetry


def _center_distance_norm(row: int, col: int, th: int, tw: int, search_shape: tuple) -> float:
    sh, sw = search_shape
    center_row, center_col = (sh - th) / 2.0, (sw - tw) / 2.0
    dist = math.hypot(row - center_row, col - center_col)
    max_dist = math.hypot(center_row, center_col)
    return float(dist / max_dist) if max_dist > 1e-6 else 0.0


def extract_candidate_features(ref: np.ndarray, search: np.ndarray,
                                 return_ncc: bool = False):
    """Run A's exact NCC+NMS front end, then compute the full feature vector
    for every candidate peak. Returns a list of dicts (one per candidate),
    sorted by descending NCC score (index 0 = A's own pure-argmax pick).

    If return_ncc=True, also returns (ncc_surface, template_shape) so a
    caller can subpixel-refine whichever candidate it ultimately picks
    (e.g. localize_hybrid.py, after the gate decides on a final index) --
    default False preserves the original single-return-value signature for
    existing callers (train_ranker.py) unchanged."""
    template = downscale_reference(ref, factor=ZOOM_RATIO)
    th, tw = template.shape
    ncc = fft_ncc(search, template)

    min_dist = max(5, min(th, tw) // NMS_MIN_DISTANCE_DIVISOR)
    peaks = non_max_suppression(
        ncc, min_distance=min_dist,
        score_threshold=NMS_SCORE_THRESHOLD, max_peaks=MAX_PEAKS,
    )

    if not peaks:
        r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
        peaks = [type("P", (), {"row": float(r), "col": float(c),
                                 "score": float(ncc[r, c])})()]

    n_candidates = len(peaks)
    top_score = peaks[0].score
    n_candidates_norm = float(math.log1p(n_candidates))

    results = []
    for rank, p in enumerate(peaks):
        row, col = int(round(p.row)), int(round(p.col))
        window, cy, cx = _extract_window(ncc, row, col)
        curv_x, curv_y = _directional_curvature(window, cy, cx)
        basin_width, symmetry = _basin_width_and_symmetry(window, cy, cx)
        anisotropy = abs(curv_x - curv_y) / (abs(curv_x) + abs(curv_y) + 1e-9)
        center_dist = _center_distance_norm(row, col, th, tw, search.shape)

        feature_vec = np.array([
            p.score, float(rank), top_score - p.score,
            curv_x, curv_y, symmetry, anisotropy,
            basin_width, center_dist, n_candidates_norm,
        ], dtype=np.float32)

        results.append(dict(
            row=row, col=col, score=float(p.score), rank=rank,
            pred_x=col + tw / 2.0, pred_y=row + th / 2.0,
            features=feature_vec,
        ))

    if return_ncc:
        return results, ncc, (th, tw)
    return results
