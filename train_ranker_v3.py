"""
train_ranker_v3.py
======================
Drift-Sense hackathon -- V3 candidate ranker.

Evaluation reports THREE different metrics:

1. Candidate coverage
   Fraction of ALL pairs where at least one generated candidate
   is within the requested tolerance of GT.

2. Conditional top-1
   Ranking accuracy among only the eligible pairs where a valid
   candidate exists.

3. End-to-end accuracy
   Correctly localized pairs / ALL pairs.

Relationship:

    end_to_end = candidate_coverage * conditional_top1


Usage:
    python train_ranker_v3.py \
        --dataset-dir ./center_biased_dataset \
        --tolerance 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from extended_features import (
    extract_extended_features,
    EXTENDED_FEATURE_NAMES,
    DEFAULT_TOP_K,
)



SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def resolve_path(dataset_dir: str, path: str) -> str:
    """Robust to metadata paths that are absolute, already-correct
    relative, or relative-to-dataset-dir -- avoids the class of bug this
    session hit before (a path getting resolved relative to the wrong
    base directory, e.g. accidentally doubling the dataset_dir prefix)."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(dataset_dir, path)
    if os.path.exists(candidate):
        return candidate
    # Fall back to the original path -- let the eventual file-open error
    # surface naturally rather than silently returning a guess.
    return path

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_gray(path: str) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float64,
    )


# ============================================================
# GROUPING
# ============================================================

def group_key(m: dict) -> tuple:
    """
    Groups samples using the underlying preset/sweep/bucket
    information.

    Samples sharing these values remain in the same split.
    """

    preset = m.get("noise_params", {}).get("preset_name")

    return (
        preset,
        m.get("sweep"),
        m.get("bucket"),
    )


# ============================================================
# GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def grouped_split(
    metadata: list,
    val_fraction: float,
    test_fraction: float,
    seed: int,
):
    """
    Split metadata into disjoint TRAIN / VALIDATION / TEST sets.

    Splitting happens at GROUP level rather than pair level to avoid
    leakage between closely related samples.

    Guarantees:
        - train, validation and test groups are disjoint
        - at least 3 groups remain in training
        - validation is separate from test
        - test is never used for model selection
    """

    MAX_COVERAGE_FRACTION = 0.6
    MIN_TRAIN_GROUPS = 3

    groups = defaultdict(list)

    for m in metadata:
        groups[group_key(m)].append(m)

    keys = sorted(
        groups.keys(),
        key=lambda k: str(k),
    )

    n_total_groups = len(keys)

    rng = random.Random(seed)

    if n_total_groups < MIN_TRAIN_GROUPS + 2:
        raise RuntimeError(
            f"Only {n_total_groups} total groups found -- need at least "
            f"{MIN_TRAIN_GROUPS + 2} groups to guarantee a non-trivial "
            f"train set plus validation and test."
        )

    def key_preset(k):
        return k[0]

    def key_bucket(k):
        return k[2]

    def carve_coverage_set(
        pool,
        target_n,
        max_n,
        presets_seen,
        buckets_seen,
    ):
        """
        Select groups while trying to preserve preset/bucket diversity.
        """

        pool = pool[:]
        rng.shuffle(pool)

        selected = []

        # First: introduce unseen presets.
        for k in list(pool):

            if len(selected) >= max_n:
                break

            preset = key_preset(k)

            if preset is not None and preset not in presets_seen:
                selected.append(k)
                presets_seen.add(preset)
                buckets_seen.add(key_bucket(k))
                pool.remove(k)

        # Second: introduce unseen buckets.
        for k in list(pool):

            if len(selected) >= max_n:
                break

            bucket = key_bucket(k)

            if bucket is not None and bucket not in buckets_seen:
                selected.append(k)
                buckets_seen.add(bucket)
                presets_seen.add(key_preset(k))
                pool.remove(k)

        # Finally fill to target.
        selected_set = set(selected)

        if len(selected_set) < target_n:

            n_more = target_n - len(selected_set)

            extra = [
                k
                for k in pool
                if k not in selected_set
            ][:n_more]

            selected_set |= set(extra)

        remaining_pool = [
            k
            for k in pool
            if k not in selected_set
        ]

        return selected_set, remaining_pool

    presets_seen = set()
    buckets_seen = set()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    n_test_target = max(
        1,
        int(round(n_total_groups * test_fraction)),
    )

    max_test = max(
        1,
        min(
            int(n_total_groups * MAX_COVERAGE_FRACTION),
            n_total_groups - MIN_TRAIN_GROUPS - 1,
        ),
    )

    test_keys, remaining_after_test = carve_coverage_set(
        keys,
        n_test_target,
        max_test,
        presets_seen,
        buckets_seen,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    n_val_target = max(
        1,
        int(round(len(remaining_after_test) * val_fraction)),
    )

    max_val = max(
        1,
        min(
            int(len(remaining_after_test) * MAX_COVERAGE_FRACTION),
            len(remaining_after_test) - MIN_TRAIN_GROUPS,
        ),
    )

    val_keys, remaining_after_val = carve_coverage_set(
        remaining_after_test,
        n_val_target,
        max_val,
        presets_seen,
        buckets_seen,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_keys = set(remaining_after_val)

    if not train_keys:
        raise RuntimeError(
            "Grouped split produced an empty TRAIN set."
        )

    train = []
    val = []
    test = []

    for k, members in groups.items():

        if k in test_keys:
            test.extend(members)

        elif k in val_keys:
            val.extend(members)

        else:
            train.extend(members)

    print(
        f"  Split: {n_total_groups} total groups -> "
        f"{len(train)} train pairs, "
        f"{len(val)} validation pairs, "
        f"{len(test)} test pairs"
    )

    return train, val, test


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def build_candidate_groups(
    metadata: list,
    tolerance: float,
    top_k: int,
    dataset_dir: str,
) -> list:

    groups = []

    for i, m in enumerate(metadata):

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

        # ----------------------------------------------------
        # Generate candidates + features
        # ----------------------------------------------------

        candidates, ncc, (th, tw) = extract_extended_features(
            ref,
            search,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # Calculate distance of every candidate from GT
        # ----------------------------------------------------

        errors = [
            float(
                np.hypot(
                    c["col"] + tw / 2.0 - gx,
                    c["row"] + th / 2.0 - gy,
                )
            )
            for c in candidates
        ]

        if not errors:
            raise RuntimeError(
                f"No candidates generated for pair "
                f"{m.get('pair_id')}"
            )

        best_idx = int(
            np.argmin(errors)
        )

        has_positive = (
            errors[best_idx] <= tolerance
        )

        # ----------------------------------------------------
        # Build feature matrix
        # ----------------------------------------------------

        X = np.stack(
            [
                c["features"]
                for c in candidates
            ]
        )

        # ----------------------------------------------------
        # Ranking target
        #
        # Exactly one candidate is positive.
        # Only pairs with at least one candidate within tolerance
        # are eligible for ranking accuracy.
        # ----------------------------------------------------

        y = np.zeros(
            len(candidates),
            dtype=np.float32,
        )

        if has_positive:
            y[best_idx] = 1.0

        groups.append(
            dict(
                pair_id=m.get("pair_id"),
                X=X,
                y=y,
                has_positive=has_positive,
                n_candidates=len(candidates),
            )
        )

        print(
            f"[{i+1}/{len(metadata)}] "
            f"{m.get('pair_id')}: "
            f"{len(candidates)} candidates, "
            f"has_positive={has_positive}"
        )

    return groups


# ============================================================
# NORMALIZATION
# ============================================================

def fit_normalization(groups: list):

    X_all = np.vstack(
        [
            g["X"]
            for g in groups
        ]
    )

    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0)

    sd[sd < 1e-8] = 1.0

    return (
        mu.astype(np.float32),
        sd.astype(np.float32),
    )


def apply_normalization(
    groups: list,
    mu: np.ndarray,
    sd: np.ndarray,
):

    return [
        {
            **g,
            "X": (g["X"] - mu) / sd,
        }
        for g in groups
    ]


# ============================================================
# MODEL
# ============================================================

class CandidateRanker(nn.Module):

    def __init__(
        self,
        n_features: int,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_features,
                32,
            ),

            nn.LayerNorm(32),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(
                32,
                16,
            ),

            nn.GELU(),

            nn.Dropout(0.05),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, x):

        return self.net(x).squeeze(-1)


# ============================================================
# LISTWISE LOSS
# ============================================================

def listwise_loss(
    model,
    groups: list,
    device,
):

    losses = []

    for g in groups:

        if not g["has_positive"]:
            continue

        x = torch.from_numpy(
            g["X"]
        ).to(device)

        y = torch.from_numpy(
            g["y"]
        ).to(device)

        logits = model(x)

        log_probs = F.log_softmax(
            logits,
            dim=0,
        )

        losses.append(
            -(y * log_probs).sum()
        )

    if not losses:
        raise RuntimeError(
            "No positive-containing groups in batch."
        )

    return torch.stack(losses).mean()


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_top1(
    model,
    groups,
    device,
):
    """
    Returns:

        coverage:
            fraction of ALL pairs having at least one candidate
            within tolerance.

        conditional_top1:
            ranking accuracy among eligible pairs.

        end_to_end:
            correct / ALL pairs.

    Therefore:

        end_to_end = coverage * conditional_top1
    """

    total = len(groups)

    eligible = 0
    correct = 0

    for g in groups:

        # No candidate near GT -> impossible for ranker
        if not g["has_positive"]:
            continue

        eligible += 1

        x = torch.from_numpy(
            g["X"]
        ).to(device)

        pred_idx = int(
            torch.argmax(
                model(x)
            )
        )

        if g["y"][pred_idx] > 0:
            correct += 1

    coverage = (
        eligible /
        max(1, total)
    )

    conditional_top1 = (
        correct /
        max(1, eligible)
    )

    end_to_end = (
        correct /
        max(1, total)
    )

    return (
        coverage,
        conditional_top1,
        end_to_end,
        eligible,
        correct,
    )


# ============================================================
# MAIN
# ============================================================

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
        "--tolerance",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    ap.add_argument(
        "--top-k-scale",
        type=int,
        default=DEFAULT_TOP_K,
    )

    ap.add_argument(
        "--model-out",
        default=None,
    )

    ap.add_argument(
        "--config-out",
        default=None,
    )

    args = ap.parse_args()

    # ========================================================
    # SEED
    # ========================================================

    seed_everything(
        args.seed
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Features: {EXTENDED_FEATURE_NAMES} "
        f"(top_k_scale={args.top_k_scale})\n"
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    metadata_path = os.path.join(
        args.dataset_dir,
        args.metadata_file,
    )

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(
        f"Loaded {len(metadata)} pairs "
        f"from {args.dataset_dir}\n"
    )

    # ========================================================
    # GROUPED SPLIT
    # ========================================================

    train_meta, val_meta, test_meta = grouped_split(
        metadata,
        args.val_fraction,
        args.test_fraction,
        args.seed,
    )

    # ========================================================
    # EXPLICIT LEAK CHECK
    # ========================================================

    train_keys = {
        group_key(m)
        for m in train_meta
    }

    val_keys = {
        group_key(m)
        for m in val_meta
    }

    test_keys = {
        group_key(m)
        for m in test_meta
    }

    leak_train_val = (
        train_keys & val_keys
    )

    leak_train_test = (
        train_keys & test_keys
    )

    leak_val_test = (
        val_keys & test_keys
    )

    if (
        leak_train_val
        or leak_train_test
        or leak_val_test
    ):
        raise RuntimeError(
            f"Group leakage detected -- "
            f"train/val: {leak_train_val} "
            f"train/test: {leak_train_test} "
            f"val/test: {leak_val_test}"
        )

    print(
        f"Train: {len(train_meta)} pairs  "
        f"Validation: {len(val_meta)} pairs  "
        f"Test: {len(test_meta)} pairs"
    )

    print(
        "Leak check "
        "(all 3 pairwise combinations): PASS\n"
    )

    # ========================================================
    # BUILD TRAIN CANDIDATES
    # ========================================================

    print(
        "--- Building TRAIN candidates "
        "(slower than v1/v2 -- scale feature) ---"
    )

    train_groups = build_candidate_groups(
        train_meta,
        args.tolerance,
        args.top_k_scale,
        args.dataset_dir,
    )

    n_train_positive = sum(
        1
        for g in train_groups
        if g["has_positive"]
    )

    if n_train_positive == 0:
        raise RuntimeError(
            "Training set contains zero "
            "positive-containing groups."
        )

    # ========================================================
    # BUILD VALIDATION CANDIDATES
    # ========================================================

    print(
        "\n--- Building VALIDATION candidates ---"
    )

    val_groups = build_candidate_groups(
        val_meta,
        args.tolerance,
        args.top_k_scale,
        args.dataset_dir,
    )

    n_val_positive = sum(
        1
        for g in val_groups
        if g["has_positive"]
    )

    if n_val_positive == 0:
        raise RuntimeError(
            "Validation set contains zero "
            "positive-containing groups."
        )

    # ========================================================
    # BUILD TEST CANDIDATES
    # ========================================================

    print(
        "\n--- Building TEST candidates ---"
    )

    test_groups = build_candidate_groups(
        test_meta,
        args.tolerance,
        args.top_k_scale,
        args.dataset_dir,
    )

    # ========================================================
    # NORMALIZATION
    #
    # IMPORTANT:
    # Fit ONLY on TRAIN.
    # ========================================================

    mu, sd = fit_normalization(
        train_groups
    )

    train_groups = apply_normalization(
        train_groups,
        mu,
        sd,
    )

    val_groups = apply_normalization(
        val_groups,
        mu,
        sd,
    )

    test_groups = apply_normalization(
        test_groups,
        mu,
        sd,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = CandidateRanker(
        n_features=len(
            EXTENDED_FEATURE_NAMES
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_top1 = -1.0
    best_epoch = 0
    stale = 0
    best_state = None

    print(
        "\n--- Training "
        "(model selection on VALIDATION, "
        "TEST untouched) ---"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        order = list(
            range(
                len(train_groups)
            )
        )

        random.shuffle(
            order
        )

        shuffled = [
            train_groups[i]
            for i in order
        ]

        optimizer.zero_grad()

        loss = listwise_loss(
            model,
            shuffled,
            device,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # TRAIN / VALIDATION EVALUATION
        # ----------------------------------------------------

        model.eval()

        (
            train_coverage,
            train_top1,
            train_e2e,
            _,
            _,
        ) = evaluate_top1(
            model,
            train_groups,
            device,
        )

        (
            val_coverage,
            val_top1,
            val_e2e,
            n_val_eligible,
            _,
        ) = evaluate_top1(
            model,
            val_groups,
            device,
        )

        # ----------------------------------------------------
        # MODEL SELECTION
        #
        # VALIDATION ONLY.
        # TEST NEVER TOUCHED HERE.
        # ----------------------------------------------------

        if val_top1 > best_val_top1:

            best_val_top1 = val_top1
            best_epoch = epoch
            stale = 0

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in model.state_dict().items()
            }

        else:

            stale += 1

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == best_epoch
        ):

            print(
                f"epoch={epoch:3d} "
                f"loss={loss.item():.5f} "
                f"train_top1={train_top1*100:5.1f}% "
                f"val_top1={val_top1*100:5.1f}% "
                f"val_cov={val_coverage*100:5.1f}% "
                f"val_e2e={val_e2e*100:5.1f}%"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if stale >= args.patience:

            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(best was epoch {best_epoch})"
            )

            break

    # ========================================================
    # RESTORE BEST VALIDATION CHECKPOINT
    # ========================================================

    if best_state is None:
        raise RuntimeError(
            "No best model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # ========================================================
    # FINAL TRAIN METRICS
    # ========================================================

    (
        final_train_coverage,
        final_train_top1,
        final_train_e2e,
        n_train_eligible,
        n_train_correct,
    ) = evaluate_top1(
        model,
        train_groups,
        device,
    )

    # ========================================================
    # FINAL VALIDATION METRICS
    # ========================================================

    (
        final_val_coverage,
        final_val_top1,
        final_val_e2e,
        n_val_eligible,
        n_val_correct,
    ) = evaluate_top1(
        model,
        val_groups,
        device,
    )

    # ========================================================
    # FINAL TEST METRICS
    #
    # IMPORTANT:
    # This is the FIRST time test results are evaluated.
    # ========================================================

    (
        final_test_coverage,
        final_test_top1,
        final_test_e2e,
        n_test_eligible,
        n_test_correct,
    ) = evaluate_top1(
        model,
        test_groups,
        device,
    )

    if n_test_eligible == 0:

        raise RuntimeError(
            "Test set contains zero eligible groups. "
            "Candidate coverage is 0%, so conditional ranking "
            "accuracy cannot be meaningfully evaluated."
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Features: {EXTENDED_FEATURE_NAMES}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTRAIN"
    )

    print(
        f"  Candidate coverage:      "
        f"{final_train_coverage*100:.2f}% "
        f"({n_train_eligible}/{len(train_groups)})"
    )

    print(
        f"  Conditional top-1:       "
        f"{final_train_top1*100:.2f}% "
        f"({n_train_correct}/{n_train_eligible})"
    )

    print(
        f"  End-to-end accuracy:     "
        f"{final_train_e2e*100:.2f}% "
        f"({n_train_correct}/{len(train_groups)})"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nVALIDATION"
    )

    print(
        f"  Candidate coverage:      "
        f"{final_val_coverage*100:.2f}% "
        f"({n_val_eligible}/{len(val_groups)})"
    )

    print(
        f"  Conditional top-1:       "
        f"{final_val_top1*100:.2f}% "
        f"({n_val_correct}/{n_val_eligible})"
    )

    print(
        f"  End-to-end accuracy:     "
        f"{final_val_e2e*100:.2f}% "
        f"({n_val_correct}/{len(val_groups)})"
    )

    print(
        "  Used for model selection: YES"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nTEST"
    )

    print(
        f"  Candidate coverage:      "
        f"{final_test_coverage*100:.2f}% "
        f"({n_test_eligible}/{len(test_groups)})"
    )

    print(
        f"  Conditional top-1:       "
        f"{final_test_top1*100:.2f}% "
        f"({n_test_correct}/{n_test_eligible})"
    )

    print(
        f"  End-to-end accuracy:     "
        f"{final_test_e2e*100:.2f}% "
        f"({n_test_correct}/{len(test_groups)})"
    )

    print(
        "  Used for model selection: NO"
    )

    print(
        "  Evaluated after model freeze: YES"
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    calculated_e2e = (
        final_test_coverage
        * final_test_top1
    )

    print(
        "\nTEST METRIC CONSISTENCY CHECK"
    )

    print(
        f"  coverage × conditional top-1: "
        f"{calculated_e2e*100:.2f}%"
    )

    print(
        f"  reported end-to-end:           "
        f"{final_test_e2e*100:.2f}%"
    )

    if not np.isclose(
        calculated_e2e,
        final_test_e2e,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Metric consistency check failed."
        )

    print(
        "  PASS"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        args.model_out
        or os.path.join(
            args.dataset_dir,
            "ranker_v3.pt",
        )
    )

    config_path = (
        args.config_out
        or os.path.join(
            args.dataset_dir,
            "ranker_v3_config.json",
        )
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "n_features":
                len(EXTENDED_FEATURE_NAMES),

            "feature_names":
                EXTENDED_FEATURE_NAMES,

            "top_k_scale":
                args.top_k_scale,

            "mean":
                mu.tolist(),

            "std":
                sd.tolist(),

            "best_epoch":
                best_epoch,

            "best_val_top1":
                best_val_top1,

            "final_test_candidate_coverage":
                final_test_coverage,

            "final_test_conditional_top1":
                final_test_top1,

            "final_test_end_to_end":
                final_test_e2e,
        },
        model_path,
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            dict(

                feature_names=
                    EXTENDED_FEATURE_NAMES,

                top_k_scale=
                    args.top_k_scale,

                mean=
                    mu.tolist(),

                std=
                    sd.tolist(),

                tolerance=
                    args.tolerance,

                val_fraction=
                    args.val_fraction,

                test_fraction=
                    args.test_fraction,

                seed=
                    args.seed,

                best_epoch=
                    best_epoch,

                best_val_top1=
                    best_val_top1,

                # Train
                train_candidate_coverage=
                    final_train_coverage,

                train_conditional_top1=
                    final_train_top1,

                train_end_to_end=
                    final_train_e2e,

                # Validation
                val_candidate_coverage=
                    final_val_coverage,

                val_conditional_top1=
                    final_val_top1,

                val_end_to_end=
                    final_val_e2e,

                # Test
                final_test_candidate_coverage=
                    final_test_coverage,

                final_test_conditional_top1=
                    final_test_top1,

                final_test_end_to_end=
                    final_test_e2e,

                n_train_pairs=
                    len(train_meta),

                n_val_pairs=
                    len(val_meta),

                n_test_pairs=
                    len(test_meta),

                n_train_eligible=
                    n_train_eligible,

                n_val_eligible=
                    n_val_eligible,

                n_test_eligible=
                    n_test_eligible,

                n_train_correct=
                    n_train_correct,

                n_val_correct=
                    n_val_correct,

                n_test_correct=
                    n_test_correct,
            ),
            f,
            indent=2,
        )

    # ========================================================
    # DONE
    # ========================================================

    print(
        f"\nSaved model:  {model_path}"
    )

    print(
        f"Saved config: {config_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()