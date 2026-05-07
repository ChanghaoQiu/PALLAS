#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_template_learning.py — Module 1: State-Template Learning
===============================================================
Offline training phase that discovers traffic states and builds per-state
template codebooks for upload and download directions separately.

Pipeline position
-----------------
  train.npz  →  [State-Template Learning]  →  stl_config.pkl

Steps
-----
1. Slide a K-slot window over every training trace to compute per-slot
   upload and download arrival counts and cumulative load features (Lambda).
2. Run K-Means on Lambda values to discover M traffic states; derive
   midpoint thresholds between adjacent cluster centres.
3. For each state, build a discrete template codebook over per-slot packet
   counts (upload and download learned independently).
4. Learn a global (state-agnostic) reference codebook for each direction,
   used downstream by the Template Smoothing module.

Input
-----
* ``train.npz`` in WFLib NPZ format.

Output
------
* ``stl_config.pkl`` — pickle file (passed to template_smoothing.py).
* ``stl_config.json`` — human-readable JSON snapshot.

Usage
-----
::

    python state_template_learning.py \\
        --train_npz ./data/train.npz \\
        --out       ./config/stl_config.pkl \\
        --delta 0.2 --K_slot 30 --num_states 2
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from utils import load_npz_timeseries, normalize, config_to_jsonable


# ── Window construction ───────────────────────────────────────────────────────

def build_windows_for_training(
    times:  np.ndarray,
    dirs:   np.ndarray,
    delta:  float,
    K_slot: int,
) -> Tuple[List[float], List[float], List[float], List[float], List[int], List[int]]:
    """Compute per-slot counts and K-slot sliding-window load (Lambda) for
    every slot in every training trace.

    Returns flattened lists (one entry per slot across all traces):
        Lambda_total, Lambda_up, Lambda_down, loads_total, us, ds
    """
    K = int(max(K_slot, 0))
    all_Lambda_total: List[float] = []
    all_Lambda_up:    List[float] = []
    all_Lambda_down:  List[float] = []
    all_loads:        List[float] = []
    all_us:           List[int]   = []
    all_ds:           List[int]   = []

    for i in range(times.shape[0]):
        t_seq, d_seq = times[i], dirs[i]
        mask = t_seq != 0
        if not np.any(mask):
            continue

        t_nz    = t_seq[mask];  d_nz = d_seq[mask]
        start_t = float(t_nz[0])
        dur     = float(t_nz[-1]) - start_t
        n_slots = 1 if dur <= 0 else int(np.floor(dur / delta)) + 1

        up = np.zeros(n_slots, np.int32)
        dn = np.zeros(n_slots, np.int32)
        sidx = np.clip(
            np.floor((t_nz - start_t) / delta).astype(np.int64), 0, n_slots - 1)
        for s, d in zip(sidx, d_nz):
            if d > 0:   up[s] += 1
            elif d < 0: dn[s] += 1

        tot  = up + dn
        p_t  = np.cumsum(tot)
        p_u  = np.cumsum(up)
        p_d  = np.cumsum(dn)

        for t in range(n_slots):
            if K <= 0:
                lam_t = lam_u = lam_d = 0.0
            else:
                lam_t = float(p_t[t] - p_t[t-K]) if t >= K else float(p_t[t])
                lam_u = float(p_u[t] - p_u[t-K]) if t >= K else float(p_u[t])
                lam_d = float(p_d[t] - p_d[t-K]) if t >= K else float(p_d[t])
            all_Lambda_total.append(lam_t)
            all_Lambda_up.append(lam_u)
            all_Lambda_down.append(lam_d)
            all_loads.append(float(tot[t]))
            all_us.append(int(up[t]))
            all_ds.append(int(dn[t]))

    return all_Lambda_total, all_Lambda_up, all_Lambda_down, all_loads, all_us, all_ds


# ── State discovery ───────────────────────────────────────────────────────────

def choose_fixed_states(
    Lambda_list:  List[float],
    num_states:   int,
    sample_size:  int = 20000,
    random_state: int = 0,
) -> Tuple[int, List[float], np.ndarray]:
    """Cluster Lambda values into traffic states using K-Means.

    Returns (K, thresholds, centers_sorted).
    """
    arr = np.asarray(Lambda_list, np.float64)
    n   = len(arr)
    if n < 2 or num_states <= 1:
        return 1, [], np.array([float(arr.mean()) if n else 0.0])

    K  = min(int(num_states), n)
    X  = arr.reshape(-1, 1)
    eff = max(sample_size, K)
    Xs  = X[np.random.choice(n, min(n, eff), replace=False)]

    km_init = KMeans(n_clusters=K, random_state=random_state, n_init="auto").fit(Xs)
    km_full = KMeans(n_clusters=K, random_state=random_state,
                     init=km_init.cluster_centers_, n_init=1).fit(X)

    ctrs = np.sort(km_full.cluster_centers_.flatten())
    thrs = [float(0.5 * (ctrs[i] + ctrs[i+1])) for i in range(K-1)]
    return K, thrs, ctrs


def assign_states_from_thresholds(
    Lambda_arr: np.ndarray,
    thresholds: List[float],
) -> np.ndarray:
    """Map each Lambda value to a state index using pre-computed thresholds."""
    arr = np.asarray(Lambda_arr, np.float64)
    ids = np.zeros(len(arr), np.int32)
    for i, lam in enumerate(arr):
        s = 0
        for thr in thresholds:
            if lam >= thr:
                s += 1
        ids[i] = s
    return ids


# ── Codebook construction ─────────────────────────────────────────────────────

def _codebook_from_samples(
    samples:       np.ndarray,
    max_templates: int,
    random_state:  int = 0,
) -> Dict[str, np.ndarray]:
    X = np.asarray(samples)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Ns = len(X)
    if Ns == 0:
        return {"targets": np.array([[0]], np.int32), "probs": np.array([1.0])}

    if Ns <= max_templates:
        uniq, cnts = np.unique(X, axis=0, return_counts=True)
        return {"targets": uniq.astype(np.int32),
                "probs":   (cnts / float(Ns)).astype(np.float64)}

    km    = KMeans(n_clusters=max_templates, random_state=random_state, n_init="auto").fit(X)
    cnts  = np.bincount(km.labels_, minlength=max_templates).astype(np.float64)
    tgts  = np.maximum(np.rint(km.cluster_centers_).astype(np.int32), 0)
    uniq, inv = np.unique(tgts, axis=0, return_inverse=True)
    merged    = np.zeros(len(uniq), np.float64)
    for k, p in enumerate(cnts / float(Ns)):
        merged[inv[k]] += p
    return {"targets": uniq.astype(np.int32), "probs": normalize(merged)}


def learn_state_codebook_1d(
    samples_1d:    List[int],
    state_ids:     np.ndarray,
    num_states:    int,
    max_templates: int,
    random_state:  int = 0,
) -> List[Dict[str, np.ndarray]]:
    """Build per-state codebooks for one traffic direction."""
    arr  = np.asarray(samples_1d, np.int32)
    sids = np.asarray(state_ids,  np.int32)
    K    = int(num_states)
    cbs: List[Optional[Dict]] = [None] * K

    for s in range(K):
        idx = np.where(sids == s)[0]
        if idx.size:
            cbs[s] = _codebook_from_samples(arr[idx], max_templates, random_state)

    for s in range(K):
        if cbs[s] is None:
            for nb in ([cbs[s-1]] if s > 0 else []) + ([cbs[s+1]] if s+1 < K else []):
                if nb is not None:
                    cbs[s] = {"targets": nb["targets"].copy(), "probs": nb["probs"].copy()}
                    break
            if cbs[s] is None:
                cbs[s] = {"targets": np.array([[0]], np.int32), "probs": np.array([1.0])}

    return [cb for cb in cbs if cb is not None]


def learn_global_codebook_1d(
    samples_1d:           List[int],
    max_global_templates: int,
    random_state:         int = 0,
) -> Dict[str, np.ndarray]:
    """Build a state-agnostic reference codebook (used by Template Smoothing)."""
    return _codebook_from_samples(
        np.asarray(samples_1d, np.int32).reshape(-1),
        max_global_templates, random_state)


# ── Top-level function ────────────────────────────────────────────────────────

def run_state_template_learning(
    train_times:          np.ndarray,
    train_dirs:           np.ndarray,
    delta:                float,
    K_slot:               int,
    num_states:           int,
    max_templates:        int,
    max_global_templates: int,
    random_state:         int = 0,
    kmeans_sample_size:   int = 20000,
) -> dict:
    """Run State-Template Learning on training data.

    Parameters
    ----------
    train_times, train_dirs  : 2-D arrays (N × L)
    delta                    : time-slot width in seconds
    K_slot                   : sliding-window width for load (Lambda) computation
    num_states               : number of traffic states M
    max_templates            : max entries per per-state codebook
    max_global_templates     : max entries in the global reference codebook
    random_state             : random seed for K-Means
    kmeans_sample_size       : max traces sampled for K-Means (0 = no limit)

    Returns
    -------
    stl_config : dict  (passed directly to template_smoothing.run_template_smoothing)
    """
    np.random.seed(int(random_state))

    N = train_times.shape[0]
    ks = int(kmeans_sample_size) if kmeans_sample_size > 0 else N
    if 0 < ks < N:
        sel = np.random.choice(N, ks, replace=False)
        t_sub, d_sub = train_times[sel], train_dirs[sel]
        print(f"[STL] Subsampled {N} → {ks} traces for K-Means")
    else:
        t_sub, d_sub = train_times, train_dirs
        print(f"[STL] Using all {N} traces")

    _, L_up, L_dn, _, us, ds = build_windows_for_training(t_sub, d_sub, delta, K_slot)
    print(f"[STL] {len(us)} windows  (delta={delta}, K_slot={K_slot})")

    # Upload
    K_up, thr_up, ctr_up = choose_fixed_states(L_up, num_states, random_state=random_state)
    ids_up  = assign_states_from_thresholds(np.array(L_up), thr_up)
    cnts_up = np.bincount(ids_up, minlength=K_up).astype(np.int64)
    cbs_up  = learn_state_codebook_1d(us, ids_up, K_up, max_templates, random_state)
    g_up    = learn_global_codebook_1d(us, max_global_templates, random_state)
    print(f"[STL] Upload   states={K_up}  thresholds={[round(t,2) for t in thr_up]}")

    # Download
    K_dn, thr_dn, ctr_dn = choose_fixed_states(L_dn, num_states, random_state=random_state)
    ids_dn  = assign_states_from_thresholds(np.array(L_dn), thr_dn)
    cnts_dn = np.bincount(ids_dn, minlength=K_dn).astype(np.int64)
    cbs_dn  = learn_state_codebook_1d(ds, ids_dn, K_dn, max_templates, random_state)
    g_dn    = learn_global_codebook_1d(ds, max_global_templates, random_state)
    print(f"[STL] Download states={K_dn}  thresholds={[round(t,2) for t in thr_dn]}")

    return {
        "delta":               float(delta),
        "K_slot":              int(K_slot),
        "requested_num_states":int(num_states),
        "num_states_up":       int(K_up),
        "thresholds_up":       [float(x) for x in thr_up],
        "centers_up":          ctr_up.astype(np.float64),
        "state_counts_up":     cnts_up.tolist(),
        "codebooks_up":        cbs_up,
        "global_codebook_up":  g_up,
        "num_states_down":     int(K_dn),
        "thresholds_down":     [float(x) for x in thr_dn],
        "centers_down":        ctr_dn.astype(np.float64),
        "state_counts_down":   cnts_dn.tolist(),
        "codebooks_down":      cbs_dn,
        "global_codebook_down":g_dn,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="State-Template Learning")
    p.add_argument("--train_npz",            required=True)
    p.add_argument("--out",                  default="./config/stl_config.pkl")
    p.add_argument("--delta",                type=float, default=0.2)
    p.add_argument("--K_slot",               type=int,   default=30)
    p.add_argument("--num_states",           type=int,   default=2)
    p.add_argument("--max_templates",        type=int,   default=48)
    p.add_argument("--max_global_templates", type=int,   default=64)
    p.add_argument("--seed",                 type=int,   default=42)
    p.add_argument("--kmeans_sample_size",   type=int,   default=20000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[STL] Loading {args.train_npz}")
    times, dirs, _ = load_npz_timeseries(args.train_npz)

    config = run_state_template_learning(
        train_times          = times,
        train_dirs           = dirs,
        delta                = args.delta,
        K_slot               = args.K_slot,
        num_states           = args.num_states,
        max_templates        = args.max_templates,
        max_global_templates = args.max_global_templates,
        random_state         = args.seed,
        kmeans_sample_size   = args.kmeans_sample_size,
    )

    with open(args.out, "wb") as f:
        pickle.dump(config, f)
    print(f"[STL] Saved → {args.out}")

    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump(config_to_jsonable(config), f, indent=2)
    print(f"[STL] JSON  → {json_out}")


if __name__ == "__main__":
    main()
