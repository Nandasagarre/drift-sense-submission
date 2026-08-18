"""
localize_a.py
=============
Drift-Sense hackathon -- VARIANT A: full-image FFT-NCC baseline.

Given a reference (high-res) image and a wide-search (low-res) image, known to differ
by exactly the problem statement's fixed 10x zoom ratio:
    1. Downscale the reference by 10x to match the search image's pixel scale.
    2. Run zero-mean NCC across the ENTIRE search image in one shot (via FFT).
    3. Non-max suppress the correlation surface to find ALL strong candidate peaks
       (periodic DRAM/FinFET patterns produce more than one near-identical peak).
    4. Select the single highest-scoring peak (argmax) -- see SELECTION RULE note.
    5. Subpixel-refine the winning peak.


    dataset                     argmax    tie-break   net effect
    full_ablation_v3 (n=300)    78.7%     74.7%       -12 pairs
    dataset/ (confounded, n=200) 17.0%    13.0%        -8 pairs
    ablation_dataset_v2 (n=150)  78.0%    72.7%        -8 pairs
    center_biased_dataset (n=150) 80.0%   76.7%        -5 pairs (best case for tie-break)
    ablation_dataset v1 (n=170)   5.9%     5.9%        tied (jitter-off, both unsolvable)

Usage:
    python localize_a.py --reference ref.png --search search.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from matching import (
    downscale_reference, fft_ncc, non_max_suppression, confidence_score,
    subpixel_refine, LocalizationResult,
)

ZOOM_RATIO = 10  # fixed by the problem statement: reference is captured at 10x the
                  # resolution of the wide-search image.

from scipy.ndimage import gaussian_filter






def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def localize(ref: np.ndarray, search: np.ndarray,
             nms_min_distance: int = 5, nms_threshold: float = 0.1) -> LocalizationResult:
    """Full-image FFT-NCC localization (Variant A). `ref` is the FULL-RESOLUTION
    reference array (will be downscaled internally by ZOOM_RATIO); `search` is the
    wide-search array at its native resolution."""
    t0 = time.perf_counter()

    template = downscale_reference(ref, factor=ZOOM_RATIO)
    th, tw = template.shape

    

    ncc = fft_ncc(search, template)

    # min_distance for NMS should be roughly the template size itself, since two
    # peaks closer than that can't both be genuinely distinct matches (they'd
    # overlap the same search-image pixels).
    min_dist = max(nms_min_distance, min(th, tw) // 3)
    peaks = non_max_suppression(ncc, min_distance=min_dist, score_threshold=nms_threshold)

    if not peaks:
        # Nothing cleared the threshold -- fall back to the single best offset
        # regardless of score, rather than returning no answer at all.
        r, c = np.unravel_index(np.argmax(ncc), ncc.shape)
        peaks = [type("P", (), {"row": float(r), "col": float(c), "score": float(ncc[r, c])})()]

    conf = confidence_score(peaks)

    # SELECTION RULE: pure argmax -- peaks is already sorted descending by score,
    # so peaks[0] is simply the single highest-scoring candidate anywhere in the
    # search image. See module docstring for why this replaced the earlier
    # center-distance tie-break.
    best = peaks[0]

    r_sub, c_sub = subpixel_refine(ncc, int(round(best.row)), int(round(best.col)))

    # Convert top-left-offset coordinate -> match CENTER coordinate, in search-image
    # pixel space (0,0 = top-left, per the organizer's stated CSV convention).
    pred_x = c_sub + tw / 2.0
    pred_y = r_sub + th / 2.0

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return LocalizationResult(
        pred_x=pred_x, pred_y=pred_y, confidence=conf,
        peak1=peaks[0].score, peak2=(peaks[1].score if len(peaks) > 1 else None),
        runtime_ms=runtime_ms, windows_evaluated=1,
        final_window_size=search.shape, early_exit=False, fallback=False,
        estimated_pitch_px=None, num_candidates=len(peaks),
    )


def main():
    ap = argparse.ArgumentParser(description="Variant A: full-image FFT-NCC localization.")
    ap.add_argument("--reference", required=True, help="Path to the reference image.")
    ap.add_argument("--search", required=True, help="Path to the wide-search image.")
    args = ap.parse_args()

    ref = load_gray(args.reference)
    search = load_gray(args.search)
    result = localize(ref, search)

    print(f"pred_x={result.pred_x:.3f}, pred_y={result.pred_y:.3f}")
    print(f"confidence={result.confidence}, peak1={result.peak1:.4f}, peak2={result.peak2}")
    print(f"runtime_ms={result.runtime_ms:.2f}, num_candidates={result.num_candidates}")


if __name__ == "__main__":
    main()

