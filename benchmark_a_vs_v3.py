"""
benchmark_a_vs_v3.py
====================

Drift-Sense -- A vs V3 benchmark.

Methods:
    A       -> baseline only
    V3      -> V3 only
    both    -> A + V3 comparison

Examples:

A only:
    python benchmark_a_vs_v3.py \
        --dataset-dir ./center_biased_dataset \
        --tolerance 5.0 \
        --out a_results.csv \
        --test-only \
        --method A

V3 only:
    python benchmark_a_vs_v3.py \
        --dataset-dir ./center_biased_dataset \
        --model ./center_biased_dataset/ranker_v3.pt \
        --config ./center_biased_dataset/ranker_v3_config.json \
        --tolerance 5.0 \
        --out v3_results.csv \
        --test-only \
        --method V3

Both:
    python benchmark_a_vs_v3.py \
        --dataset-dir ./center_biased_dataset \
        --model ./center_biased_dataset/ranker_v3.pt \
        --config ./center_biased_dataset/ranker_v3_config.json \
        --tolerance 5.0 \
        --out a_vs_v3_results.csv \
        --test-only \
        --method both
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats

import numpy as np
import torch
from PIL import Image

import localize_a
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
# PROCESS A
# ============================================================================

def run_a(ref, search, gx, gy, tolerance):

    ra = localize_a.localize(
        ref,
        search,
    )

    err = error_px(
        ra.pred_x,
        ra.pred_y,
        gx,
        gy,
    )

    return {
        "A_pred_x": ra.pred_x,
        "A_pred_y": ra.pred_y,
        "A_error_px": err,
        "A_correct": err <= tolerance,
    }


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
# PROCESS ONE PAIR
# ============================================================================

def process_one(
    m,
    dataset_dir,
    method,
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

    # ------------------------------------------------------------------------
    # A
    # ------------------------------------------------------------------------

    if method in ("A", "both"):

        row.update(
            run_a(
                ref,
                search,
                gx,
                gy,
                tolerance,
            )
        )

    # ------------------------------------------------------------------------
    # V3
    # ------------------------------------------------------------------------

    if method in ("V3", "both"):

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
# A VS V3 COMPARISON
# ============================================================================

def print_comparison(rows, tolerance):

    n = len(rows)

    a_acc = sum(
        r["A_correct"]
        for r in rows
    ) / n

    v3_acc = sum(
        r["V3_correct"]
        for r in rows
    ) / n

    a_errors = [
        r["A_error_px"]
        for r in rows
    ]

    v3_errors = [
        r["V3_error_px"]
        for r in rows
    ]

    print()
    print("=" * 80)
    print("A VS V3")
    print("=" * 80)

    print(
        f"A :  {a_acc*100:6.2f}%  "
        f"mean={stats.mean(a_errors):.2f}px  "
        f"median={stats.median(a_errors):.2f}px"
    )

    print(
        f"V3:  {v3_acc*100:6.2f}%  "
        f"mean={stats.mean(v3_errors):.2f}px  "
        f"median={stats.median(v3_errors):.2f}px"
    )

    print(
        f"\nV3 - A accuracy: "
        f"{(v3_acc-a_acc)*100:+.2f} percentage points"
    )

    # ------------------------------------------------------------------------
    # Recoveries / regressions
    # ------------------------------------------------------------------------

    recoveries = [
        r
        for r in rows
        if not r["A_correct"]
        and r["V3_correct"]
    ]

    regressions = [
        r
        for r in rows
        if r["A_correct"]
        and not r["V3_correct"]
    ]

    print()
    print("RECOVERIES / REGRESSIONS")
    print("-" * 80)

    print(
        f"Recoveries : "
        f"{len(recoveries)}"
    )

    print(
        f"Regressions: "
        f"{len(regressions)}"
    )

    # ------------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------------

    changed = sum(
        r["changed"]
        for r in rows
    )

    disagreed = sum(
        r["disagreed"]
        for r in rows
    )

    print()
    print("V3 GATE")
    print("-" * 80)

    print(
        f"Disagreed: "
        f"{disagreed}/{n} "
        f"({disagreed/n*100:.1f}%)"
    )

    print(
        f"Override accepted: "
        f"{changed}/{n} "
        f"({changed/n*100:.1f}%)"
    )

    # ------------------------------------------------------------------------
    # Gate reasons
    # ------------------------------------------------------------------------

    reasons = sorted(
        set(
            r["gate_reason"]
            for r in rows
        )
    )

    print()
    print("BY GATE REASON")
    print("-" * 80)

    for reason in reasons:

        sub = [
            r
            for r in rows
            if r["gate_reason"] == reason
        ]

        if not sub:
            continue

        a = (
            sum(r["A_correct"] for r in sub)
            / len(sub)
        )

        v3 = (
            sum(r["V3_correct"] for r in sub)
            / len(sub)
        )

        print(
            f"{reason:30s} "
            f"n={len(sub):4d} "
            f"A={a*100:6.2f}% "
            f"V3={v3*100:6.2f}%"
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
        choices=["A", "V3", "both"],
        default="both",
        help="Run A, V3, or both",
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

    if args.method in ("V3", "both"):

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
        if args.method in ("V3", "both"):

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

        else:

            # A-only:
            # If metadata already contains split information,
            # select test directly.
            test_metadata = [
                m
                for m in metadata
                if str(m.get("split", "")).lower()
                == "test"
            ]

            if test_metadata:

                metadata = test_metadata

                print(
                    f"--test-only: "
                    f"{len(metadata)} TEST pairs from metadata split"
                )

            else:

                print(
                    "--test-only requested, but metadata has no "
                    "'split=test' entries; evaluating all pairs."
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
            method=args.method,
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

        if args.method == "A":

            msg = (
                f"A_error={row['A_error_px']:.3f}px "
                f"{'OK' if row['A_correct'] else 'FAIL'}"
            )

        elif args.method == "V3":

            msg = (
                f"V3_error={row['V3_error_px']:.3f}px "
                f"{'OK' if row['V3_correct'] else 'FAIL'}"
            )

        else:

            msg = (
                f"A={row['A_error_px']:.3f}px "
                f"V3={row['V3_error_px']:.3f}px"
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

    if args.method == "A":

        print_single_summary(
            rows,
            "A",
            args.tolerance,
        )

    elif args.method == "V3":

        print_single_summary(
            rows,
            "V3",
            args.tolerance,
        )

    else:

        print_comparison(
            rows,
            args.tolerance,
        )


if __name__ == "__main__":
    main()