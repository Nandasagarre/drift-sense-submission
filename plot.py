"""
plot.py
-------
Generate diagnostic plots from the V3 benchmark CSV.

Usage:
    python plot.py --csv benchmark_results.csv

The script is intentionally defensive:
- A plot is generated only if the required columns/data exist.
- Missing columns or insufficient data are skipped with a clear message.
- No plot failure stops the remaining plots.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[SKIP] Missing columns: {', '.join(missing)}")
        return False
    return True


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def save_plot(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]   {path}")


def clean_xy(df, x, y):
    tmp = df[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    return tmp.dropna()


# ---------------------------------------------------------------------------
# 1. PR curve per noise level
# ---------------------------------------------------------------------------

def plot_pr_by_noise(df, outdir):
    required = ["baseline_score", "V3_correct", "noise_level"]

    if not has_cols(df, required):
        return

    data = df[required].copy()
    data["score"] = pd.to_numeric(data["baseline_score"], errors="coerce")
    data["correct"] = data["V3_correct"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    data["noise_level"] = data["noise_level"].fillna("unknown").astype(str)
    data = data.dropna(subset=["score"])

    if len(data) < 2:
        print("[SKIP] PR curves: insufficient rows.")
        return

    # A simple score-threshold PR calculation.
    # Positive = V3 localization is correct within the benchmark tolerance.
    fig, ax = plt.subplots(figsize=(8, 6))

    plotted = 0

    for noise, g in data.groupby("noise_level", sort=True):
        if len(g) < 2:
            print(f"[SKIP] PR curve for noise={noise}: fewer than 2 samples.")
            continue

        y = g["correct"].astype(bool).to_numpy()
        scores = g["score"].to_numpy(dtype=float)

        positives = int(y.sum())
        negatives = len(y) - positives

        if positives == 0:
            print(f"[SKIP] PR curve for noise={noise}: no positive samples.")
            continue

        if negatives == 0:
            print(
                f"[SKIP] PR curve for noise={noise}: "
                "no negative samples; precision is always 1."
            )
            continue

        thresholds = np.unique(scores)[::-1]

        precision = []
        recall = []

        for threshold in thresholds:
            pred = scores >= threshold

            tp = np.sum(pred & y)
            fp = np.sum(pred & ~y)
            fn = np.sum(~pred & y)

            p = tp / (tp + fp) if (tp + fp) else 1.0
            r = tp / (tp + fn) if (tp + fn) else 0.0

            precision.append(p)
            recall.append(r)

        # Add endpoints.
        recall = np.asarray([0.0] + recall + [1.0])
        precision = np.asarray([1.0] + precision + [positives / len(y)])

        ax.plot(
            recall,
            precision,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{noise} (n={len(g)})",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("[SKIP] No valid noise group for PR curve.")
        return

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("V3 Precision–Recall Curve by Noise Level")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    save_plot(fig, outdir, "01_pr_curve_by_noise.png")


# ---------------------------------------------------------------------------
# 2. Baseline score by noise level
# ---------------------------------------------------------------------------

def plot_score_by_noise(df, outdir):
    if not has_cols(df, ["baseline_score", "noise_level"]):
        return

    data = df[["baseline_score", "noise_level"]].copy()
    data["baseline_score"] = pd.to_numeric(
        data["baseline_score"], errors="coerce"
    )
    data["noise_level"] = data["noise_level"].fillna("unknown").astype(str)
    data = data.dropna(subset=["baseline_score"])

    if data.empty:
        print("[SKIP] Baseline score by noise: no numeric score data.")
        return

    groups = []
    labels = []

    for noise, g in data.groupby("noise_level", sort=True):
        if len(g):
            groups.append(g["baseline_score"].to_numpy())
            labels.append(noise)

    if not groups:
        print("[SKIP] Baseline score by noise: no usable groups.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        groups,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Baseline confidence score")
    ax.set_title("Baseline Score by Noise Level")
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=30, ha="right")

    save_plot(fig, outdir, "02_baseline_score_by_noise.png")


# ---------------------------------------------------------------------------
# 3. V3 error distribution
# ---------------------------------------------------------------------------

def plot_error_distribution(df, outdir):
    if not has_cols(df, ["V3_error_px"]):
        return

    errors = numeric(df, "V3_error_px").dropna()

    if len(errors) < 2:
        print("[SKIP] Error distribution: insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(errors, bins="auto")
    ax.set_xlabel("V3 localization error (px)")
    ax.set_ylabel("Number of image pairs")
    ax.set_title("V3 Localization Error Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, outdir, "03_v3_error_distribution.png")


# ---------------------------------------------------------------------------
# 4. Error vs confidence
# ---------------------------------------------------------------------------

def plot_error_vs_score(df, outdir):
    if not has_cols(df, ["baseline_score", "V3_error_px"]):
        return

    data = clean_xy(df, "baseline_score", "V3_error_px")

    if len(data) < 2:
        print("[SKIP] Error vs score: insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        data["baseline_score"],
        data["V3_error_px"],
        s=45,
        alpha=0.75,
    )

    ax.set_xlabel("Baseline confidence score")
    ax.set_ylabel("V3 localization error (px)")
    ax.set_title("V3 Error vs Baseline Confidence")
    ax.grid(True, alpha=0.3)

    save_plot(fig, outdir, "04_error_vs_baseline_score.png")


# ---------------------------------------------------------------------------
# 5. Accuracy by noise level
# ---------------------------------------------------------------------------

def plot_accuracy_by_noise(df, outdir):
    if not has_cols(df, ["noise_level", "V3_correct"]):
        return

    data = df[["noise_level", "V3_correct"]].copy()
    data["noise_level"] = data["noise_level"].fillna("unknown").astype(str)
    data["correct"] = data["V3_correct"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    summary = data.groupby("noise_level")["correct"].agg(
        ["mean", "count"]
    )

    if summary.empty:
        print("[SKIP] Accuracy by noise: no usable data.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(summary))

    ax.bar(x, summary["mean"] * 100)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_ylabel("V3 accuracy (%)")
    ax.set_xlabel("Noise level")
    ax.set_title("V3 Accuracy by Noise Level")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            i,
            row["mean"] * 100,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_plot(fig, outdir, "05_accuracy_by_noise.png")


# ---------------------------------------------------------------------------
# 6. Runtime per image pair
# ---------------------------------------------------------------------------

def plot_runtime_by_noise(df, outdir):
    if not has_cols(df, ["noise_level", "time_per_image_pair_ms"]):
        return

    data = df[["noise_level", "time_per_image_pair_ms"]].copy()
    data["noise_level"] = data["noise_level"].fillna("unknown").astype(str)
    data["time_ms"] = pd.to_numeric(
        data["time_per_image_pair_ms"], errors="coerce"
    )
    data = data.dropna(subset=["time_ms"])

    if data.empty:
        print("[SKIP] Runtime by noise: no timing data.")
        return

    summary = data.groupby("noise_level")["time_ms"].agg(
        ["mean", "median", "count"]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(summary))

    ax.bar(x, summary["mean"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("Time per image pair (ms)")
    ax.set_title("V3 Runtime by Noise Level")
    ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, outdir, "06_runtime_by_noise.png")


# ---------------------------------------------------------------------------
# 7. Spatial error vectors
# ---------------------------------------------------------------------------

def plot_error_vectors(df, outdir):
    if not has_cols(df, ["V3_dx_px", "V3_dy_px"]):
        return

    data = clean_xy(df, "V3_dx_px", "V3_dy_px")

    if len(data) < 2:
        print("[SKIP] Error vectors: insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        data["V3_dx_px"],
        data["V3_dy_px"],
        s=45,
        alpha=0.75,
    )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_xlabel("Prediction error Δx (px)")
    ax.set_ylabel("Prediction error Δy (px)")
    ax.set_title("V3 Spatial Localization Error")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    save_plot(fig, outdir, "07_spatial_error_vectors.png")


# ---------------------------------------------------------------------------
# 8. Error vs rotation
# ---------------------------------------------------------------------------

def plot_error_by_rotation(df, outdir):
    if not has_cols(df, ["polygon_rotation_deg", "V3_error_px"]):
        return

    data = df[["polygon_rotation_deg", "V3_error_px"]].copy()
    data["rotation"] = pd.to_numeric(
        data["polygon_rotation_deg"], errors="coerce"
    )
    data["error"] = pd.to_numeric(
        data["V3_error_px"], errors="coerce"
    )
    data = data.dropna()

    if len(data) < 2:
        print("[SKIP] Error by rotation: insufficient data.")
        return

    groups = []
    labels = []

    for rotation, g in data.groupby("rotation", sort=True):
        groups.append(g["error"].to_numpy())
        tick_labels.append(str(rotation))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_xlabel("Polygon rotation (°)")
    ax.set_ylabel("V3 localization error (px)")
    ax.set_title("V3 Error by Polygon Rotation")
    ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, outdir, "08_error_by_rotation.png")


# ---------------------------------------------------------------------------
# 9. Accuracy by architecture
# ---------------------------------------------------------------------------

def plot_accuracy_by_architecture(df, outdir):
    if not has_cols(df, ["architecture", "V3_correct"]):
        return

    data = df[["architecture", "V3_correct"]].copy()
    data["architecture"] = data["architecture"].fillna("unknown").astype(str)
    data["correct"] = data["V3_correct"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    summary = data.groupby("architecture")["correct"].agg(
        ["mean", "count"]
    )

    if summary.empty:
        print("[SKIP] Accuracy by architecture: no usable data.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(summary))
    ax.bar(x, summary["mean"] * 100)

    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("V3 accuracy (%)")
    ax.set_title("V3 Accuracy by DRAM Architecture")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    save_plot(fig, outdir, "09_accuracy_by_architecture.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="V3 benchmark CSV",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="Directory for generated plots",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=" * 70)
    print("V3 BENCHMARK PLOTS")
    print("=" * 70)
    print(f"CSV      : {csv_path}")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print(f"Output   : {outdir}")
    print("=" * 70)

    # Core requested plots first.
    plot_pr_by_noise(df, outdir)
    plot_score_by_noise(df, outdir)

    # Basic diagnostic plots.
    plot_error_distribution(df, outdir)
    plot_error_vs_score(df, outdir)
    plot_accuracy_by_noise(df, outdir)
    plot_runtime_by_noise(df, outdir)
    plot_error_vectors(df, outdir)
    plot_error_by_rotation(df, outdir)
    plot_accuracy_by_architecture(df, outdir)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
