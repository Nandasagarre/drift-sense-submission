"""
benchmark_v3.py
====================

Drift-Sense -- V3-only benchmark.

Method:
    V3      -> V3 localization only

Example:
    V3 only:
    python benchmark_a_vs_v3.py \
        --dataset-dir ./center_biased_dataset \
        --model ./center_biased_dataset/ranker_v3.pt \
        --config ./center_biased_dataset/ranker_v3_config.json \
        --tolerance 5.0 \
        --out v3_results.csv \
        --test-only \
        --method V3

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
import time

import numpy as np
import torch
from PIL import Image

import localize_best_v3
from train_ranker_v3 import grouped_split


# ============================================================================
# HELPERS
# ============================================================================

def resolve_path(dataset_dir: str, path: str) -> str:
    if path is None:
        return path

    if os.path.isabs(path) and os.path.exists(path):
        return path

    if os.path.exists(path):
        return path

    candidate = os.path.join(dataset_dir, path)

    if os.path.exists(candidate):
        return candidate

    return path


def load_gray(path: str):
    return np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float64,
    )


def error_px(px, py, gx, gy):
    return float(
        ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
    )


# ============================================================================
# DYNAMIC METADATA FLATTENER
# ============================================================================

def flatten_metadata(obj, parent_key="", sep="."):
    """
    Flatten arbitrary nested JSON.

    Dict:
        noise_params.gamma

    List:
        gt_center_px.0
        gt_center_px.1

    Any future metadata fields automatically appear in CSV.
    """

    items = {}

    if isinstance(obj, dict):

        for key, value in obj.items():

            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else str(key)
            )

            items.update(
                flatten_metadata(
                    value,
                    new_key,
                    sep,
                )
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            new_key = (
                f"{parent_key}{sep}{i}"
                if parent_key
                else str(i)
            )

            items.update(
                flatten_metadata(
                    value,
                    new_key,
                    sep,
                )
            )

    else:
        items[parent_key] = obj

    return items


# ============================================================================
# PROCESS V3
# ============================================================================

def run_v3(
    ref,
    search,
    gx,
    gy,
    model,
    mu,
    sd,
    top_k_scale,
    device,
    tolerance,
):

    rv3 = localize_best_v3.localize(
        ref,
        search,
        model,
        mu,
        sd,
        top_k_scale,
        device,
    )

    # Final gated V3 prediction
    err_v3 = error_px(
        rv3["pred_x"],
        rv3["pred_y"],
        gx,
        gy,
    )

    # Raw ranker prediction
    ranker_err = error_px(
        rv3["ranker_pred_x"],
        rv3["ranker_pred_y"],
        gx,
        gy,
    )

    return {
        "V3_pred_x": rv3["pred_x"],
        "V3_pred_y": rv3["pred_y"],
        "V3_error_px": err_v3,
        "V3_correct": err_v3 <= tolerance,

        "gate_reason": rv3["gate_reason"],
        "changed": rv3["changed"],
        "disagreed": rv3["disagreed"],

        "ranker_pred_x": rv3["ranker_pred_x"],
        "ranker_pred_y": rv3["ranker_pred_y"],
        "ranker_error_px": ranker_err,
        "ranker_correct": ranker_err <= tolerance,

        "ncc_confidence_gap": rv3["ncc_confidence_gap"],
        "ranker_margin": rv3["ranker_margin"],
        "center_penalty": rv3["center_penalty"],
        "effective_margin": rv3["effective_margin"],
    }


# ============================================================================
# BENCHMARK / PLOTTING METRICS
# ============================================================================

def _first_numeric(d, keys):
    """Return first finite numeric value for the supplied keys."""
    for key in keys:
        value = d.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def derive_noise_level(row):
    """
    Prefer an explicit noise level if present.
    Otherwise use the existing noise preset name.

    The original metadata is left unchanged.
    """
    explicit = (
        row.get("noise_level")
        or row.get("noise_params.noise_level")
        or row.get("noise_params.level")
    )
    if explicit not in (None, ""):
        return explicit

    return row.get("noise_params.preset_name", "unknown")


def add_plot_metrics(row, tolerance):
    """
    Add stable scalar fields useful for PR curves and basic plots.
    Higher baseline_score means more confident.
    """
    row["noise_level"] = derive_noise_level(row)

    # Primary confidence score for PR analysis.
    # V3 already exports ranker_margin, effective_margin and NCC gap.
    row["baseline_score"] = _first_numeric(
        row,
        ["ranker_margin", "effective_margin", "ncc_confidence_gap"],
    )

    row["ranker_score"] = row.get("ranker_margin")
    row["ncc_score"] = row.get("ncc_confidence_gap")

    row["V3_positive"] = bool(row.get("V3_correct", False))
    row["V3_error_over_tolerance"] = (
        float(row["V3_error_px"]) > float(tolerance)
    )

    row["V3_error_ratio"] = (
        float(row["V3_error_px"]) / float(tolerance)
        if tolerance > 0 else np.nan
    )

    # Spatial error components for vector/bias plots.
    gx = float(row["gt_center_px.0"])
    gy = float(row["gt_center_px.1"])
    px = float(row["V3_pred_x"])
    py = float(row["V3_pred_y"])

    row["V3_dx_px"] = px - gx
    row["V3_dy_px"] = py - gy
    row["V3_abs_dx_px"] = abs(px - gx)
    row["V3_abs_dy_px"] = abs(py - gy)

    return row


# ============================================================================
# PROCESS ONE PAIR
# ============================================================================

def process_one(
    m,
    dataset_dir,
    model,
    mu,
    sd,
    top_k_scale,
    device,
    tolerance,
):
    ref = load_gray(
        resolve_path(
            dataset_dir,
            m["ref_path"],
        )
    )

    search = load_gray(
        resolve_path(
            dataset_dir,
            m["search_path"],
        )
    )

    gx, gy = m["gt_center_px"]

    # ALL metadata from JSON
    row = flatten_metadata(m)

    # Time only the actual V3 image-pair search/localization.
    t0 = time.perf_counter()

    row.update(
        run_v3(
            ref,
            search,
            gx,
            gy,
            model,
            mu,
            sd,
            top_k_scale,
            device,
            tolerance,
        )
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    row["search_time_ms"] = elapsed_ms
    row["time_per_image_pair_ms"] = elapsed_ms

    add_plot_metrics(row, tolerance)

    return row


# ============================================================================
# CSV
# ============================================================================

def write_csv(rows, output_path):

    fields = []
    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    return fields


# ============================================================================
# SUMMARY
# ============================================================================

def print_single_summary(rows, method, tolerance):

    n = len(rows)

    key = method

    errors = [
        r[f"{key}_error_px"]
        for r in rows
    ]

    correct = [
        r[f"{key}_correct"]
        for r in rows
    ]

    acc = sum(correct) / n

    print()
    print("=" * 80)
    print(f"{method}-ONLY BENCHMARK")
    print("=" * 80)

    print(f"Pairs       : {n}")
    print(f"Tolerance   : {tolerance:.2f}px")
    print(f"Accuracy    : {acc * 100:.2f}%")
    print(f"Mean error  : {stats.mean(errors):.2f}px")
    print(f"Median error: {stats.median(errors):.2f}px")
    print(f"Max error   : {max(errors):.2f}px")

    times_ms = [
        float(r["time_per_image_pair_ms"])
        for r in rows
        if r.get("time_per_image_pair_ms") is not None
    ]

    if times_ms:
        print(f"Mean time   : {stats.mean(times_ms):.2f} ms/pair")
        print(f"Median time : {stats.median(times_ms):.2f} ms/pair")
        print(f"Max time    : {max(times_ms):.2f} ms/pair")

    scores = [
        r.get("baseline_score")
        for r in rows
        if r.get("baseline_score") is not None
    ]

    if scores:
        print(f"Mean score  : {stats.mean(scores):.6f}")
        print(f"Median score: {stats.median(scores):.6f}")

    # ------------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------------

    print()
    print(f"{method} ACCURACY BY STYLE")
    print("-" * 80)

    styles = sorted(
        set(
            str(r.get("style", "unknown")).lower()
            for r in rows
        )
    )

    for style in styles:

        sub = [
            r
            for r in rows
            if str(r.get("style", "unknown")).lower()
            == style
        ]

        if not sub:
            continue

        sub_acc = (
            sum(
                r[f"{method}_correct"]
                for r in sub
            )
            / len(sub)
        )

        print(
            f"{style:15s} "
            f"n={len(sub):4d} "
            f"acc={sub_acc*100:6.2f}%"
        )

    # ------------------------------------------------------------------------
    # Error bins
    # ------------------------------------------------------------------------

    bins = [
        ("0-1px", 0, 1),
        ("1-5px", 1, 5),
        ("5-10px", 5, 10),
        ("10-50px", 10, 50),
        ("50-100px", 50, 100),
        (">100px", 100, float("inf")),
    ]

    print()
    print(f"{method} ERROR DISTRIBUTION")
    print("-" * 80)

    for label, lo, hi in bins:

        count = sum(
            1
            for e in errors
            if lo <= e < hi
        )

        print(
            f"{label:>10s}: "
            f"{count:5d} "
            f"({count/n*100:6.2f}%)"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-dir",
        required=True,
    )

    ap.add_argument(
        "--metadata-file",
        default="metadata.json",
    )

    ap.add_argument(
        "--model",
        default=None,
    )

    ap.add_argument(
        "--config",
        default=None,
    )

    ap.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--out",
        default="benchmark_results.csv",
    )

    ap.add_argument(
        "--test-only",
        action="store_true",
    )

    # NEW
    ap.add_argument(
        "--method",
        choices=["V3"],
        default="V3",
        help="V3-only benchmark",
    )

    args = ap.parse_args()

    # ========================================================================
    # DEVICE
    # ========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Method: {args.method}")
    print(f"Device: {device}")

    # ========================================================================
    # LOAD V3 ONLY IF NEEDED
    # ========================================================================

    model = None
    mu = None
    sd = None
    top_k_scale = None

    if args.method == "V3":

        if not args.model:
            ap.error(
                "--model is required for --method V3 or --method both"
            )

        if not args.config:
            ap.error(
                "--config is required for --method V3 or --method both"
            )

        model, mu, sd, top_k_scale = (
            localize_best_v3.load_ranker(
                args.model,
                args.config,
                device,
            )
        )

        print(
            f"top_k_scale={top_k_scale}"
        )

    # ========================================================================
    # LOAD METADATA
    # ========================================================================

    metadata_path = os.path.join(
        args.dataset_dir,
        args.metadata_file,
    )

    with open(metadata_path) as f:
        metadata = json.load(f)

    # ========================================================================
    # TEST ONLY
    # ========================================================================

    if args.test_only:

        # Only V3 needs the ranker config for grouped split.
        # For A-only, use metadata's split field if available.
        if args.method == "V3":

            with open(args.config) as f:
                config = json.load(f)

            val_fraction = config.get(
                "val_fraction",
                0.20,
            )

            _, _, metadata = grouped_split(
                metadata,
                val_fraction,
                config["test_fraction"],
                config["seed"],
            )

            print(
                f"--test-only: "
                f"{len(metadata)} held-out TEST pairs"
            )

    # ========================================================================
    # CHECK
    # ========================================================================

    n = len(metadata)

    if n == 0:
        raise RuntimeError(
            "No metadata pairs available."
        )

    print(
        f"Loaded {n} pairs"
    )

    # ========================================================================
    # RUN
    # ========================================================================

    rows = []

    for i, m in enumerate(metadata):

        row = process_one(
            m=m,
            dataset_dir=args.dataset_dir,
            model=model,
            mu=mu,
            sd=sd,
            top_k_scale=top_k_scale,
            device=device,
            tolerance=args.tolerance,
        )

        rows.append(row)

        pair_id = m.get(
            "pair_id",
            f"pair_{i}",
        )

        msg = (
            f"V3_error={row['V3_error_px']:.3f}px "
            f"{'OK' if row['V3_correct'] else 'FAIL'} "
            f"time={row['time_per_image_pair_ms']:.2f}ms"
        )

        print(
            f"[{i+1}/{n}] {pair_id} {msg}"
        )

    # ========================================================================
    # WRITE CSV
    # ========================================================================

    fields = write_csv(
        rows,
        args.out,
    )

    print()
    print(
        f"Written to {args.out}"
    )

    print(
        f"CSV columns: {len(fields)}"
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print_single_summary(
        rows,
        "V3",
        args.tolerance,
    )


if __name__ == "__main__":
    main()