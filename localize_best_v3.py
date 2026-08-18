"""
localize_best_v3.py
======================
Drift-Sense hackathon -- A + ML, v3: SAME veto-gate architecture proven
safe all session, paired with the ranker trained by train_ranker_v3.py
(original 10 features + scale_score_std).

Usage:
    python localize_best_v3.py --reference ref.png --search search.png \
        --model center_biased_dataset/ranker_v3.pt \
        --config center_biased_dataset/ranker_v3_config.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from matching import subpixel_refine
from extended_features import extract_extended_features, EXTENDED_FEATURE_NAMES, DEFAULT_TOP_K

NCC_CONFIDENCE_VETO_GAP = 0.05
MIN_RANKER_MARGIN = 0.15
CENTER_DISTANCE_PENALTY_PER_NORM_UNIT = 0.20

_CENTER_DIST_IDX = EXTENDED_FEATURE_NAMES.index("center_distance_norm")


class CandidateRanker(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def load_ranker(model_path: str, config_path: str, device):
    with open(config_path) as f:
        config = json.load(f)
    checkpoint = torch.load(model_path, map_location=device)
    model = CandidateRanker(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    mu = np.array(config["mean"], dtype=np.float32)
    sd = np.array(config["std"], dtype=np.float32)
    top_k_scale = config.get("top_k_scale", DEFAULT_TOP_K)
    return model, mu, sd, top_k_scale


def _softmax_margin(scores: np.ndarray, top_idx: int) -> float:
    sorted_scores = np.sort(scores)[::-1]
    top1 = scores[top_idx]
    top2 = sorted_scores[1] if len(sorted_scores) > 1 else -np.inf
    if not np.isfinite(top2):
        return 1.0
    exp_diff = np.exp(np.clip(top2 - top1, -50, 50))
    p_runner_up = exp_diff / (1.0 + exp_diff)
    return float(1.0 - 2.0 * p_runner_up)


def localize(ref: np.ndarray, search: np.ndarray, model, mu, sd, top_k_scale, device) -> dict:
    t0 = time.perf_counter()

    candidates, ncc, (th, tw) = extract_extended_features(ref, search, top_k=top_k_scale)
    n_candidates = len(candidates)

    X = np.stack([c["features"] for c in candidates])
    X_norm = (X - mu) / sd
    X_tensor = torch.from_numpy(X_norm.astype(np.float32)).to(device)

    with torch.no_grad():
        ranker_scores = model(X_tensor).detach().cpu().numpy()

    a_idx = 0
    ranker_idx = int(np.argmax(ranker_scores))
    disagreed = ranker_idx != a_idx

    final_idx = a_idx
    gate_reason = "no_disagreement" if not disagreed else None

    # Diagnostic fields (per review): None when there's no disagreement,
    # since the gate math below is only computed in that branch -- this
    # avoids reporting a misleading "0.0" that looks like a real value.
    ncc_confidence_gap = None
    ranker_margin = None
    center_penalty = None
    effective_margin = None
    ranker_score_top1 = float(ranker_scores[ranker_idx])
    ranker_score_top2 = float(np.partition(ranker_scores, -2)[-2]) if n_candidates > 1 else None

    if disagreed:
        ncc_top1 = candidates[0]["score"]
        ncc_top2 = candidates[1]["score"] if n_candidates > 1 else -1.0
        ncc_confidence_gap = float(ncc_top1 - ncc_top2)

        ranker_margin = _softmax_margin(ranker_scores, ranker_idx)

        a_center_dist = float(X[a_idx, _CENTER_DIST_IDX])
        ranker_center_dist = float(X[ranker_idx, _CENTER_DIST_IDX])
        center_penalty = max(0.0, ranker_center_dist - a_center_dist) * \
            CENTER_DISTANCE_PENALTY_PER_NORM_UNIT
        effective_margin = ranker_margin - center_penalty

        if ncc_confidence_gap > NCC_CONFIDENCE_VETO_GAP:
            final_idx, gate_reason = a_idx, "vetoed_ncc_confident"
        elif effective_margin < MIN_RANKER_MARGIN:
            final_idx, gate_reason = a_idx, "vetoed_ranker_indecisive"
        else:
            final_idx, gate_reason = ranker_idx, "override_accepted"

    best = candidates[final_idx]
    r_sub, c_sub = subpixel_refine(ncc, best["row"], best["col"])
    pred_x = c_sub + tw / 2.0
    pred_y = r_sub + th / 2.0

    # ALSO compute the ranker's own candidate position independently of the
    # gate's decision (point #6 in review): the benchmark's success
    # criterion is "within tolerance", not "matches the single training
    # positive" -- so checking whether the RANKER's raw pick would have
    # been correct, regardless of whether the gate accepted it, is the
    # diagnostic that actually distinguishes "ranker works, gate too
    # conservative" from "gate is correctly protecting against a bad pick".
    ranker_candidate = candidates[ranker_idx]
    ranker_r_sub, ranker_c_sub = subpixel_refine(ncc, ranker_candidate["row"],
                                                    ranker_candidate["col"])
    ranker_pred_x = ranker_c_sub + tw / 2.0
    ranker_pred_y = ranker_r_sub + th / 2.0

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "pred_x": float(pred_x), "pred_y": float(pred_y),
        "ranker_pred_x": float(ranker_pred_x), "ranker_pred_y": float(ranker_pred_y),
        "num_candidates": n_candidates,
        "a_index": a_idx, "ranker_index": ranker_idx, "final_index": final_idx,
        "disagreed": disagreed,
        "gate_reason": gate_reason, "changed": final_idx != a_idx,
        "ncc_confidence_gap": ncc_confidence_gap,
        "ranker_margin": ranker_margin,
        "center_penalty": center_penalty,
        "effective_margin": effective_margin,
        "ranker_score_top1": ranker_score_top1,
        "ranker_score_top2": ranker_score_top2,
        "runtime_ms": float(runtime_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, mu, sd, top_k_scale = load_ranker(args.model, args.config, device)
    ref = load_gray(args.reference)
    search = load_gray(args.search)

    result = localize(ref, search, model, mu, sd, top_k_scale, device)

    print()
    print("=" * 60)
    print("A + ML v3 (with scale_std, GATED) RESULT")
    print("=" * 60)
    print(f"pred_x={result['pred_x']:.3f}")
    print(f"pred_y={result['pred_y']:.3f}")
    print(f"gate_reason={result['gate_reason']}")
    print(f"changed={result['changed']}")
    print(f"num_candidates={result['num_candidates']}")
    print(f"runtime_ms={result['runtime_ms']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
