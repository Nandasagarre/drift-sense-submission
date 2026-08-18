"""
matching.py
===========

Core pipeline this module provides:
    1. downscale_reference()   -- apply the known 10x zoom ratio before matching
    2. fft_ncc()                -- zero-mean normalized cross-correlation surface,
                                    computed efficiently via FFT (numerator) + integral
                                    images (local windowed normalization), per Lewis
                                    (1995) "Fast Normalized Cross-Correlation."
    3. non_max_suppression()    -- extract ALL strong candidate peaks, not just argmax,
                                    since periodic DRAM/FinFET patterns produce multiple
                                    near-identical correlation peaks (see problem
                                    statement slide 4).
    4. confidence_score()       -- peak1/peak2 ratio, a "selectivity" metric (same
                                    concept used in the Kaggle CZII cryoET challenge's
                                    over-picking-ratio analysis).
    5. estimate_pitch()         -- autocorrelation-based lattice pitch estimate, used
                                    by localize_b.py to size its expanding windows
                                    WITHOUT reading the dataset's known ground-truth
                                    pitch (a real deployment wouldn't have that).
    6. subpixel_refine()        -- parabolic peak fit for sub-pixel precision.
    7. pick_by_center_distance() -- the problem statement's own tie-break rule: among
                                    near-tied candidates, prefer whichever is closest
                                    to the search image's center.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter


# --------------------------------------------------------------------------------------
# 1. Downscale reference by the known 10x zoom ratio
# --------------------------------------------------------------------------------------

def downscale_reference(ref: np.ndarray, factor: int = 10) -> np.ndarray:
    """Area-average downsample the reference by `factor` (default 10x, matching the
    problem statement's known, fixed zoom ratio). Box-averaging (not point-sampling)
    approximates the same finite-aperture PSF integration used when the dataset
    generator renders the wide-search image -- see dataset_generator.py's
    render_layout_antialiased() / CITATIONS.md #4."""
    h, w = ref.shape
    assert h % factor == 0 and w % factor == 0, \
        f"Reference shape {ref.shape} must be divisible by factor={factor}"
    nh, nw = h // factor, w // factor
    return ref.reshape(nh, factor, nw, factor).mean(axis=(1, 3))


# --------------------------------------------------------------------------------------
# 2. FFT-based zero-mean normalized cross-correlation (Lewis 1995 "fast NCC")
# --------------------------------------------------------------------------------------

def _integral_image(img: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row/col prepended, so window sums are a single
    O(1) lookup: window_sum = ii[r+h,c+w] - ii[r,c+w] - ii[r+h,c] + ii[r,c]."""
    ii = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(ii, ((1, 0), (1, 0)), mode="constant")


def _window_sum(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def fft_ncc(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalized cross-correlation surface between `search` (larger) and
    `template` (smaller), evaluated at every "valid" top-left offset (template fully
    inside search). Returns an array of shape
        (search.shape[0] - template.shape[0] + 1, search.shape[1] - template.shape[1] + 1)
    where entry [r, c] is NCC(search[r:r+th, c:c+tw], template), in [-1, 1].

    Implementation follows Lewis (1995), "Fast Normalized Cross-Correlation": the raw
    cross-correlation numerator is computed via FFT (fftconvolve), and the local
    per-window mean/energy needed for normalization is computed via integral images
    (summed-area tables) rather than a second sliding-window pass -- both steps are
    O(N log N) / O(N) respectively, avoiding the O(N^2 M^2) cost of naive sliding-
    window NCC.
    """
    search = search.astype(np.float64)
    template = template.astype(np.float64)
    th, tw = template.shape
    sh, sw = search.shape
    assert th <= sh and tw <= sw, \
        f"template {template.shape} must fit inside search {search.shape}"

    t_mean = template.mean()
    t0 = template - t_mean
    t_norm = np.sqrt(np.sum(t0 ** 2))
    if t_norm < 1e-8:
        # Degenerate (flat) template -- no meaningful correlation possible.
        return np.zeros((sh - th + 1, sw - tw + 1))

    # Cross-correlation numerator: correlate(search, t0) == convolve(search, flip(t0)).
    numerator = fftconvolve(search, t0[::-1, ::-1], mode="valid")

    ii_I = _integral_image(search)
    ii_I2 = _integral_image(search ** 2)
    sum_I = _window_sum(ii_I, th, tw)
    sum_I2 = _window_sum(ii_I2, th, tw)

    n = th * tw
    local_energy = sum_I2 - (sum_I ** 2) / n  # sum((I-mean)^2) over each window
    local_energy = np.maximum(local_energy, 0.0)
    denom = np.sqrt(local_energy) * t_norm

    ncc = np.zeros_like(numerator)
    valid = denom > 1e-6
    ncc[valid] = numerator[valid] / denom[valid]
    return np.clip(ncc, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# 3. Non-max suppression -- extract candidate peaks, not just the single argmax
# --------------------------------------------------------------------------------------

@dataclass
class Peak:
    row: float          # row (y) in the correlation-surface's own coordinate frame
    col: float           # col (x) in the correlation-surface's own coordinate frame
    score: float


def non_max_suppression(ncc: np.ndarray, min_distance: int, score_threshold: float = 0.0,
                         max_peaks: int = 20) -> list[Peak]:
    """Find local maxima in the NCC surface at least `min_distance` apart, above
    `score_threshold`, sorted by descending score. `min_distance` should be roughly
    the lattice pitch (in the same pixel units as `ncc`) so periodic repeats are
    reported as SEPARATE candidates rather than merged into one."""
    footprint_size = 2 * min_distance + 1
    local_max = maximum_filter(ncc, size=footprint_size, mode="constant", cval=-np.inf)
    is_peak = (ncc == local_max) & (ncc > score_threshold)
    rows, cols = np.nonzero(is_peak)
    scores = ncc[rows, cols]
    order = np.argsort(-scores)
    peaks = [Peak(row=float(rows[i]), col=float(cols[i]), score=float(scores[i]))
             for i in order[:max_peaks]]
    return peaks


# --------------------------------------------------------------------------------------
# 4. Confidence / selectivity score
# --------------------------------------------------------------------------------------

def confidence_score(peaks: list[Peak]) -> float:
    """peak1/peak2 ratio (both mapped to a positive scale first, since NCC in [-1,1]
    can be negative). Returns 1.0 (maximally ambiguous) if fewer than 2 peaks exist
    -- there's nothing to compare against, so treat it as the least-confident case is
    WRONG; instead we return a large number (very confident) when there's only one
    candidate at all, since there's no competing peak. Returns None if no peaks."""
    if len(peaks) == 0:
        return None
    if len(peaks) == 1:
        return float("inf")  # no competing peak -- maximally confident by construction
    # Shift scores to be strictly positive before taking a ratio (NCC in [-1, 1]).
    eps = 1e-6
    p1 = peaks[0].score + 1.0 + eps
    p2 = peaks[1].score + 1.0 + eps
    return float(p1 / p2)


# --------------------------------------------------------------------------------------
# 5. Autocorrelation-based lattice pitch estimate (NOT read from ground truth)
# --------------------------------------------------------------------------------------

def estimate_pitch(search: np.ndarray, min_pitch_px: int = 3, max_pitch_px: int = 40) -> float:
    """Estimate the dominant lattice pitch (in pixels) of a periodic image via its own
    2D autocorrelation: the autocorrelation of a periodic signal has secondary peaks
    at multiples of the true period. We take the strongest secondary peak along each
    axis within [min_pitch_px, max_pitch_px] and average them.

    This is deliberately NOT reading the dataset's known ground-truth pitch from
    metadata -- a real deployment wouldn't have that, so localize_b.py's window
    sizing has to work from something estimable from the image itself, same as a
    real system would.
    """
    img = search.astype(np.float64)
    img = img - img.mean()
    f = np.fft.fft2(img)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    h, w = ac.shape
    cy, cx = h // 2, w // 2

    row = ac[cy, cx + min_pitch_px: cx + max_pitch_px + 1]
    col = ac[cy + min_pitch_px: cy + max_pitch_px + 1, cx]
    if row.size == 0 or col.size == 0:
        return float(min_pitch_px)
    px = min_pitch_px + int(np.argmax(row))
    py = min_pitch_px + int(np.argmax(col))
    return float((px + py) / 2.0)


def estimate_pitch_xy(search: np.ndarray, min_pitch_px: int = 3,
                       max_pitch_px: int = 60) -> tuple[float, float]:
    """Like estimate_pitch(), but returns (pitch_x, pitch_y) SEPARATELY rather than
    averaged into one scalar. Needed for lattice-aware ranking (localize_c.py): DRAM
    and FinFET presets frequently have meaningfully different pitch along each axis
    (e.g. FinFET fin_pitch vs gate_pitch can differ by 2x), so averaging them away
    loses exactly the information a per-axis lattice consistency check needs. This is
    a pure ADDITION alongside estimate_pitch() -- localize_b.py's existing behavior
    (which uses the scalar version) is unaffected."""
    img = search.astype(np.float64)
    img = img - img.mean()
    f = np.fft.fft2(img)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    h, w = ac.shape
    cy, cx = h // 2, w // 2

    row = ac[cy, cx + min_pitch_px: cx + max_pitch_px + 1]
    col = ac[cy + min_pitch_px: cy + max_pitch_px + 1, cx]
    px = float(min_pitch_px + int(np.argmax(row))) if row.size else float(min_pitch_px)
    py = float(min_pitch_px + int(np.argmax(col))) if col.size else float(min_pitch_px)
    return px, py


# --------------------------------------------------------------------------------------
# Bright-artifact (charging) blob detector -- shared infra for Variant E, moved here
# from charging_failure_analysis.py so localize_e.py can reuse it directly rather than
# duplicating the logic. Rationale unchanged: charging blobs are large, spatially
# contiguous bright regions; normal periodic-pattern bright features are small,
# individually-tiny, and disconnected -- see charging_failure_analysis.py's original
# docstring and CITATIONS.md #10 for the physical justification.
# --------------------------------------------------------------------------------------

def detect_bright_blobs(img: np.ndarray, bright_percentile: float = 95.0,
                         min_blob_size: int = 80) -> list[tuple[float, float, int]]:
    """Returns [(centroid_row, centroid_col, size_px), ...] for every connected
    bright component above min_blob_size. Defaults match charging_failure_analysis.py's
    tuned settings (loosened from an earlier, less sensitive pass that missed a
    forced-100%-probability charging case entirely)."""
    from scipy.ndimage import label
    thresh = np.percentile(img, bright_percentile)
    bright_mask = img >= thresh
    labeled_arr, n = label(bright_mask)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labeled_arr == i)
        size = len(ys)
        if size >= min_blob_size:
            blobs.append((float(ys.mean()), float(xs.mean()), size))
    return blobs


# --------------------------------------------------------------------------------------
# 6. Subpixel refinement -- parabolic fit through the peak and its neighbors
# --------------------------------------------------------------------------------------

def subpixel_refine(ncc: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """1D parabolic interpolation along each axis independently through the peak
    pixel and its immediate neighbors, giving a fractional-pixel correction. Falls
    back to the integer peak location if it's on the surface's border (no neighbor
    on one side) or the fit is degenerate."""
    h, w = ncc.shape
    r, c = float(row), float(col)

    if 0 < row < h - 1:
        y0, y1, y2 = ncc[row - 1, col], ncc[row, col], ncc[row + 1, col]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            r = row + 0.5 * (y0 - y2) / denom

    if 0 < col < w - 1:
        x0, x1, x2 = ncc[row, col - 1], ncc[row, col], ncc[row, col + 1]
        denom = (x0 - 2 * x1 + x2)
        if abs(denom) > 1e-9:
            c = col + 0.5 * (x0 - x2) / denom

    return r, c


# --------------------------------------------------------------------------------------
# 7. Tie-break: closest to the search image's center (the problem statement's own rule)
# --------------------------------------------------------------------------------------

def pick_by_center_distance(peaks: list[Peak], center_row: float, center_col: float) -> Peak:
    """Among all candidate peaks, return whichever is closest to (center_row,
    center_col) -- the literal tie-break rule stated in the problem statement
    ("whichever is closest to the search image's center")."""
    def dist2(p: Peak) -> float:
        return (p.row - center_row) ** 2 + (p.col - center_col) ** 2
    return min(peaks, key=dist2)


def filter_near_top(peaks: list[Peak], score_epsilon: float = 0.02) -> list[Peak]:
    """Restrict to peaks within `score_epsilon` of the single best score. This MUST
    be applied before pick_by_center_distance(): the tie-break rule means "among
    candidates that are genuinely tied for best," not "among every peak that merely
    cleared the NMS threshold." Applying center-distance across the full threshold-
    passing set (rather than just the near-top-score set) will happily pick a weak,
    central, spurious peak over a strong but off-center true match -- peaks is
    assumed already sorted by descending score (as returned by
    non_max_suppression())."""
    if not peaks:
        return peaks
    top_score = peaks[0].score
    return [p for p in peaks if p.score >= top_score - score_epsilon]


# --------------------------------------------------------------------------------------
# Shared result container, used by both localize_a.py and localize_b.py
# --------------------------------------------------------------------------------------

def lattice_consistency_scores(peaks: list[Peak], pitch_x: float, pitch_y: float,
                                tol_frac: float = 0.15) -> list[float]:
    """For each peak, how many OTHER peaks sit at an offset that's an (approximate)
    integer multiple of (pitch_x, pitch_y) away from it -- weighted by those other
    peaks' own scores, so being corroborated by a strong peak counts more than being
    corroborated by a weak one. This operationalizes "which candidates are consistent
    with the detected lattice" from the locked experimental plan: a peak that lines
    up with several other strong peaks on a regular grid is more likely to be a real
    periodic repeat of the true pattern than an isolated one-off.

    Returns a list of consistency scores, same order/length as `peaks`. This does NOT
    replace the NCC score -- see localize_c.py for how the two are combined."""
    n = len(peaks)
    scores = [0.0] * n
    if pitch_x <= 0 or pitch_y <= 0:
        return scores

    tol_x = max(1.0, pitch_x * tol_frac)
    tol_y = max(1.0, pitch_y * tol_frac)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dr = peaks[j].row - peaks[i].row
            dc = peaks[j].col - peaks[i].col
            # Nearest integer multiple of the lattice pitch along each axis.
            k_r = round(dr / pitch_y) if pitch_y > 0 else 0
            k_c = round(dc / pitch_x) if pitch_x > 0 else 0
            resid_r = abs(dr - k_r * pitch_y)
            resid_c = abs(dc - k_c * pitch_x)
            if resid_r <= tol_y and resid_c <= tol_x:
                # Weight by the corroborating peak's own score, shifted positive
                # (NCC can be negative) so weak/negative peaks contribute ~nothing.
                scores[i] += max(0.0, peaks[j].score)
    return scores


@dataclass
class LocalizationResult:
    pred_x: float
    pred_y: float
    confidence: float
    peak1: float
    peak2: float = None
    runtime_ms: float = 0.0
    windows_evaluated: int = 1
    final_window_size: tuple = None
    early_exit: bool = False
    fallback: bool = False
    estimated_pitch_px: float = None
    num_candidates: int = 0
    search_radius_px: float = None   # half-width of the window the ACCEPTED decision
                                      # was made from -- None for Variant A (whole image,
                                      # no windowing concept applies)
    pitch_x_px: float = None         # Variant C only: per-axis estimated pitch
    pitch_y_px: float = None
    lattice_consistency_score: float = None  # Variant C only: winning peak's own score
    lattice_pool_size: int = None            # Variant C only: size of the near-top pool
                                              # lattice-ranking was applied within
    gate_used_curvature: bool = False        # Variant F only: whether the veto+anisotropy
                                              # gate selected the curvature candidate over
                                              # the raw NCC winner for this pair
    near_eq_candidate_count: int = None      # Variant F only: size of the near-equal
                                              # candidate pool the gate actually reasons
                                              # over (NOT the same as num_candidates, which
                                              # is the full NMS peak list at a much looser
                                              # threshold -- exposed separately because the
                                              # MAX_CANDIDATES_FOR_GATE guard needs the real
                                              # count it operates on, not a proxy for it)
