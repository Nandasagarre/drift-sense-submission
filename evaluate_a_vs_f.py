"""
evaluate_a_vs_f.py
=====================
Drift-Sense hackathon -- reproduces the organizer's own baseline_solution/evaluate.py
plots (Precision-Recall curves + AP/accuracy trend), but comparing Variant A vs
Variant F instead of their noise-level sweep. Uses the EXACT same PR/AP methodology
as their script: score-threshold sweep, trapezoidal AP, precision/recall arrays
anchored at (0,1).

If the dataset has 'sweep'/'bucket' fields, also produces a SECOND set of plots
broken down by bucket for Variant A -- a direct structural match to their own
"PR by noise level" plot, using whatever condition axis this dataset varies.

Usage:
    python evaluate_a_vs_f.py --dataset-dir ./full_ablation_v3 --tolerance-px 5.0
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import localize_a
#import localize_f

TOLERANCE_DEFAULT = 5.0


def load_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float64)


def error_px(px, py, gx, gy) -> float:
    return float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)


def pr_curve(scores, corrects, n_total):
    order = np.argsort(-np.asarray(scores))
    corrects_sorted = np.asarray(corrects)[order]

    tp_cum = np.cumsum(corrects_sorted)
    fp_cum = np.cumsum(~corrects_sorted)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / max(n_total, 1)

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def average_precision(precision, recall):
    order = np.argsort(recall)
    # FIX: getattr's default arg is evaluated eagerly, not lazily -- the original
    # `getattr(np, "trapezoid", np.trapz)` pattern crashes on any numpy version
    # missing BOTH names (e.g. new enough to have dropped trapz, in an environment
    # without trapezoid for some reason), since np.trapz itself gets evaluated
    # regardless of whether trapezoid exists. Explicit hasattr check avoids this.
    if hasattr(np, "trapezoid"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    return float(trapezoid(precision[order], recall[order]))


def collect_results(metadata: list, tolerance: float) -> tuple:
    results = {"A": {"scores": [], "corrects": []}, "F": {"scores": [], "corrects": []}}
    buckets = []

    for i, m in enumerate(metadata):
        ref = load_gray(m["ref_path"])
        search = load_gray(m["search_path"])
        gx, gy = m["gt_center_px"]

        ra = localize_a.localize(ref, search)
        #rf = localize_f.localize(ref, search)

        err_a = error_px(ra.pred_x, ra.pred_y, gx, gy)
        #err_f = error_px(rf.pred_x, rf.pred_y, gx, gy)

        results["A"]["scores"].append(ra.peak1)
        results["A"]["corrects"].append(err_a <= tolerance)
        results["F"]["scores"].append(rf.peak1 if rf.peak1 is not None else ra.peak1)
        results["F"]["corrects"].append(err_f <= tolerance)

        buckets.append(m.get("bucket"))
        print(f"[{i+1}/{len(metadata)}] {m.get('pair_id')}")

    return results, buckets


def plot_pr_by_variant(results: dict, n_total: int, output_dir: str, tolerance: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    for variant, color in [("A", "tab:blue"), ("F", "tab:orange")]:
        scores = results[variant]["scores"]
        corrects = np.array(results[variant]["corrects"], dtype=bool)
        precision, recall = pr_curve(scores, corrects, n_total)
        ap = average_precision(precision, recall)
        ax.plot(recall, precision, marker="o", markersize=3, color=color,
                 label=f"{variant} (AP={ap:.2f})")
        print(f"  Variant {variant}: AP={ap:.3f}  accuracy@{tolerance}px="
              f"{corrects.mean():.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"A vs F: Precision-Recall (tol={tolerance}px)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, "pr_curve_a_vs_f.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


def plot_pr_by_bucket(results: dict, buckets: list, output_dir: str,
                       tolerance: float, variant: str = "A"):
    unique_buckets = sorted(set(b for b in buckets if b))
    if not unique_buckets:
        print("No sweep/bucket fields in this dataset -- skipping by-bucket plots.")
        return

    scores_all = np.array(results[variant]["scores"])
    corrects_all = np.array(results[variant]["corrects"], dtype=bool)
    buckets_arr = np.array(buckets)

    fig, ax = plt.subplots(figsize=(6, 5))
    ap_by_bucket, acc_by_bucket = {}, {}
    for b in unique_buckets:
        mask = buckets_arr == b
        scores_b = scores_all[mask]
        corrects_b = corrects_all[mask]
        n_b = mask.sum()
        precision, recall = pr_curve(scores_b, corrects_b, n_b)
        ap = average_precision(precision, recall)
        ap_by_bucket[b] = ap
        acc_by_bucket[b] = float(corrects_b.mean())
        ax.plot(recall, precision, marker="o", markersize=3, label=f"{b} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Variant {variant}: Precision-Recall by condition (tol={tolerance}px)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, f"pr_curve_by_bucket_{variant}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)

    labels = unique_buckets
    aps = [ap_by_bucket[b] for b in labels]
    accs = [acc_by_bucket[b] for b in labels]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax2.plot(x, aps, marker="o", label="Average Precision")
    ax2.plot(x, accs, marker="s", label=f"Accuracy (<= {tolerance}px)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Score")
    ax2.set_title(f"Variant {variant}: quality vs condition")
    ax2.legend()
    ax2.grid(alpha=0.3)
    trend_path = os.path.join(output_dir, f"ap_vs_condition_{variant}.png")
    fig2.savefig(trend_path, dpi=150, bbox_inches="tight")
    print(f"wrote {trend_path}")
    plt.close(fig2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--metadata-file", default="metadata.json")
    ap.add_argument("--tolerance-px", type=float, default=TOLERANCE_DEFAULT)
    ap.add_argument("--output-dir", default="./eval_results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset_dir, args.metadata_file)) as f:
        metadata = json.load(f)
    n_total = len(metadata)
    print(f"Loaded {n_total} pairs")

    results, buckets = collect_results(metadata, args.tolerance_px)

    print("\n--- PR curve: A vs F ---")
    plot_pr_by_variant(results, n_total, args.output_dir, args.tolerance_px)

    print("\n--- PR curve by bucket (Variant A) ---")
    plot_pr_by_bucket(results, buckets, args.output_dir, args.tolerance_px, variant="A")


if __name__ == "__main__":
    main()
