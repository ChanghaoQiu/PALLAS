#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py
================
Partition a single-file WF dataset (.npz) into train / valid / test splits
and write each split as a separate .npz file.

Split ratio : 81% train  /  9% valid  /  10% test
              (achieved by two successive 90/10 splits)

Usage
-----
::

    python datasets/split_dataset.py --dataset CW
    python datasets/split_dataset.py --dataset closed_5tab --no_stratify
"""

import argparse
import os
import sys

import numpy as np


SEED        = 2024
TRAIN_RATIO = 0.9   # first split:  90% train+valid / 10% test
VALID_RATIO = 0.9   # second split: 90% train       / 10% valid


def split_indices(n: int, ratio: float, seed: int, labels=None):
    """Return (idx_major, idx_minor) with a stratified or random 80/20-style split."""
    rng = np.random.default_rng(seed)

    if labels is not None:
        # Stratified: sample proportionally within each class
        classes = np.unique(labels)
        major_idx, minor_idx = [], []
        for c in classes:
            c_idx = np.where(labels == c)[0]
            rng.shuffle(c_idx)
            cut = max(1, round(len(c_idx) * ratio))
            major_idx.extend(c_idx[:cut].tolist())
            minor_idx.extend(c_idx[cut:].tolist())
        return np.array(major_idx), np.array(minor_idx)
    else:
        idx = rng.permutation(n)
        cut = max(1, round(n * ratio))
        return idx[:cut], idx[cut:]


def main():
    parser = argparse.ArgumentParser(description="Split a WF .npz dataset into train/valid/test.")
    parser.add_argument("--dataset",      required=True,
                        help="Dataset name, e.g. CW or closed_5tab")
    parser.add_argument("--data_dir",     default="./datasets",
                        help="Root directory that contains <dataset>.npz (default: ./datasets)")
    parser.add_argument("--no_stratify",  action="store_true",
                        help="Disable stratified splitting (use random split instead)")
    args = parser.parse_args()

    src_file  = os.path.join(args.data_dir, f"{args.dataset}.npz")
    out_dir   = os.path.join(args.data_dir, args.dataset)

    if not os.path.isfile(src_file):
        print(f"[ERROR] Source file not found: {src_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {src_file}")
    archive = np.load(src_file, allow_pickle=True)
    X, y    = archive["X"], archive["y"]

    n_classes = len(np.unique(y))
    assert n_classes == int(y.max()) + 1, \
        f"Labels are not contiguous (found {n_classes} unique values, max={y.max()})"
    print(f"[INFO] Loaded  X={X.shape}  y={y.shape}  classes={n_classes}")

    use_stratify = not args.no_stratify

    # ── First split: (train + valid) vs test ──────────────────────────────────
    tv_idx, test_idx = split_indices(
        len(y), TRAIN_RATIO, seed=SEED,
        labels=y if use_stratify else None)

    # ── Second split: train vs valid ──────────────────────────────────────────
    y_tv = y[tv_idx]
    tr_idx, val_idx = split_indices(
        len(y_tv), VALID_RATIO, seed=SEED + 1,
        labels=y_tv if use_stratify else None)

    train_idx = tv_idx[tr_idx]
    valid_idx = tv_idx[val_idx]

    # ── Save ──────────────────────────────────────────────────────────────────
    splits = {
        "train": (train_idx, "train.npz"),
        "valid": (valid_idx, "valid.npz"),
        "test":  (test_idx,  "test.npz"),
    }

    for name, (idx, fname) in splits.items():
        out_path = os.path.join(out_dir, fname)
        np.savez(out_path, X=X[idx], y=y[idx])
        print(f"[{name:5s}] X={X[idx].shape}  y={y[idx].shape}  → {out_path}")

    print(f"\n[OK] Splits written to {out_dir}/")


if __name__ == "__main__":
    main()