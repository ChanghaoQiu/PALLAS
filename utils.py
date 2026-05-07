#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========
Shared data-loading and configuration utilities.
"""

import numpy as np
from typing import Dict

EPS_TIME_SHIFT = 1e-6


def sort_timeseries_by_timestamp_inplace(times: np.ndarray, dirs: np.ndarray) -> int:
    """Stable in-place sort of each trace's non-zero packets by timestamp.
    Returns the number of traces that were re-ordered.
    """
    if times.ndim != 2 or dirs.ndim != 2 or times.shape != dirs.shape:
        raise ValueError("times and dirs must be 2-D arrays of equal shape.")
    N, L = times.shape
    fixed = 0
    for i in range(N):
        t_row, d_row = times[i], dirs[i]
        mask = t_row != 0
        k = int(mask.sum())
        if k <= 1:
            continue
        t_nz = t_row[mask]
        if not np.any(t_nz[1:] < t_nz[:-1]):
            continue
        d_nz = d_row[mask]
        order = np.argsort(t_nz, kind="mergesort")
        t_row[:k] = t_nz[order]
        d_row[:k] = d_nz[order]
        if k < L:
            t_row[k:] = 0.0
            d_row[k:] = 0
        fixed += 1
    return fixed


def load_npz_timeseries(path: str):
    """Load a WF dataset from .npz; return (times, dirs, labels).

    Supported formats
    -----------------
    ``X``-format   : ``X`` (signed; abs=timestamp, sign=direction), ``y``
    Split-format   : ``time``, ``direction``, ``y`` / ``label``
    """
    data = np.load(path, allow_pickle=True)
    keys = set(data.keys())

    if "X" in keys:
        X     = data["X"]
        times = np.abs(X).astype(np.float64)
        dirs  = np.sign(X).astype(np.int8)
        y     = data["y"]
    elif "direction" in keys and "time" in keys:
        times = data["time"].astype(np.float64)
        dirs  = data["direction"].astype(np.int8)
        y     = data["y"] if "y" in keys else data["label"]
    else:
        raise ValueError(f"Unsupported npz format: {keys}")

    fixed = sort_timeseries_by_timestamp_inplace(times, dirs)
    if fixed:
        print(f"[load] {path}: sorted {fixed}/{times.shape[0]} non-monotonic traces")
    return times, dirs, y


def normalize(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Return a valid probability vector (all positive, sum = 1)."""
    p = np.maximum(np.asarray(p, dtype=np.float64), eps)
    s = float(p.sum())
    return p / s if s > 0 else np.ones_like(p) / len(p)


def _cb_to_json(cb: dict) -> dict:
    return {"targets": np.asarray(cb["targets"]).tolist(),
            "probs":   np.asarray(cb["probs"]).tolist()}


def config_to_jsonable(config: dict) -> dict:
    """Convert a config dict to a JSON-serialisable representation."""
    cb_up   = [_cb_to_json(c) for c in config.get("codebooks_up",   [])]
    cb_down = [_cb_to_json(c) for c in config.get("codebooks_down", [])]
    g_up    = config.get("global_codebook_up",
                         {"targets": np.array([[0]]), "probs": np.array([1.0])})
    g_down  = config.get("global_codebook_down",
                         {"targets": np.array([[0]]), "probs": np.array([1.0])})
    return {
        "delta":                float(config["delta"]),
        "K_slot":               int(config["K_slot"]),
        "requested_num_states": int(config.get("requested_num_states", 1)),
        "num_states_up":        int(config.get("num_states_up",  len(cb_up)   or 1)),
        "thresholds_up":        [float(x) for x in config.get("thresholds_up",  [])],
        "centers_up":           np.asarray(config.get("centers_up",  [])).tolist(),
        "state_counts_up":      [int(x) for x in config.get("state_counts_up",  [])],
        "num_states_down":      int(config.get("num_states_down", len(cb_down) or 1)),
        "thresholds_down":      [float(x) for x in config.get("thresholds_down", [])],
        "centers_down":         np.asarray(config.get("centers_down", [])).tolist(),
        "state_counts_down":    [int(x) for x in config.get("state_counts_down", [])],
        "codebooks_up":         cb_up,
        "codebooks_down":       cb_down,
        "global_codebook_up":   _cb_to_json(g_up),
        "global_codebook_down": _cb_to_json(g_down),
        "smoothing":            config.get("smoothing", {}),
        "state_stats_up":       config.get("state_stats_up",  []),
        "state_stats_down":     config.get("state_stats_down", []),
        "tradeoff":             float(config.get("tradeoff",             0.5)),
        "idle_slots_no_real":   int(config.get("idle_slots_no_real",     2)),
        "rayleigh_scale_ratio": float(config.get("rayleigh_scale_ratio", 0.3)),
        "eps_time_shift":       float(config.get("eps_time_shift", EPS_TIME_SHIFT)),
        "drain_X":              int(config.get("drain_X", 5)),
    }
