#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
template_smoothing.py — Module 2: Template Smoothing
=====================================================
Offline post-processing that blurs per-state codebook boundaries so that
adjacent states share overlapping template distributions.

Pipeline position
-----------------
  stl_config.pkl  →  [Template Smoothing]  →  smoothed_config.pkl

How it works
------------
For each state *s* and direction (upload / download):

1. Project the global reference codebook and neighbouring codebooks onto
   state *s*'s template support via a Gaussian kernel.
2. Form a weighted mixture ``q_mix`` from the projected distributions.
3. Interpolate between the original and the mixture::

       p_new = (1 - α) · p_original  +  α · q_mix

   Larger α → stronger smoothing (states overlap more).

Input
-----
* ``stl_config.pkl`` — output of state_template_learning.py.

Output
------
* ``smoothed_config.pkl`` — pickle file (passed to online_regularization.py).
* ``smoothed_config.json`` — human-readable JSON snapshot.

Usage
-----
::

    python template_smoothing.py \\
        --config ./config/stl_config.pkl \\
        --out    ./config/smoothed_config.pkl \\
        --alpha 0.4 --sigma 1.3
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from utils import normalize, config_to_jsonable


# ── Projection ────────────────────────────────────────────────────────────────

def project_codebook_to_support(
    src_targets: np.ndarray,
    src_probs:   np.ndarray,
    dst_targets: np.ndarray,
    method:      str   = "soft",
    sigma:       float = 1.0,
) -> np.ndarray:
    """Project a source distribution onto a target support.

    Parameters
    ----------
    src_targets, src_probs : source codebook
    dst_targets            : target template support
    method                 : ``"soft"`` (Gaussian kernel) or ``"nearest"``
    sigma                  : kernel bandwidth (for ``"soft"``)

    Returns
    -------
    q : probability vector over dst_targets
    """
    G  = np.asarray(src_targets, np.float64)
    S  = np.asarray(dst_targets, np.float64)
    gp = normalize(np.asarray(src_probs, np.float64))
    if G.ndim == 1: G = G.reshape(-1, 1)
    if S.ndim == 1: S = S.reshape(-1, 1)
    M = len(S)
    if M <= 1:
        return np.ones(max(M, 1)) / max(M, 1)

    dist2 = ((G[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)

    if method == "nearest":
        q = np.zeros(M, np.float64)
        for gi, sj in enumerate(np.argmin(dist2, axis=1)):
            q[sj] += gp[gi]
        return normalize(q)

    if method != "soft":
        raise ValueError(f"Unknown projection method: {method!r}")
    sig2 = max(float(sigma), 1e-6) ** 2
    w    = np.exp(-dist2 / (2.0 * sig2))
    w   /= w.sum(axis=1, keepdims=True) + 1e-15
    return normalize((gp[:, None] * w).sum(axis=0))


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_state_codebooks(
    codebooks:       List[Dict],
    global_codebook: Dict,
    alpha:           float,
    proj:            str   = "soft",
    proj_sigma:      float = 1.0,
) -> Tuple[List[Dict], Dict]:
    """Blend each state's codebook with a mixture of its neighbours and the
    global reference distribution.

    Returns (smoothed_codebooks, smoothing_info_dict).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    K     = len(codebooks)
    if K <= 1 or alpha <= 0:
        return codebooks, {"enabled": 0.0}

    gT, gP = global_codebook["targets"], global_codebook["probs"]
    new_cbs = []

    for s, cb in enumerate(codebooks):
        T, p  = cb["targets"], normalize(cb["probs"])
        q_mix = project_codebook_to_support(gT, gP, T, proj, proj_sigma)
        w_sum = 1.0

        if s > 0:
            nb     = codebooks[s-1]
            q_mix += 0.5 * project_codebook_to_support(nb["targets"], nb["probs"], T, proj, proj_sigma)
            w_sum += 0.5
        if s + 1 < K:
            nb     = codebooks[s+1]
            q_mix += 0.5 * project_codebook_to_support(nb["targets"], nb["probs"], T, proj, proj_sigma)
            w_sum += 0.5

        q_mix = normalize(q_mix / max(w_sum, 1e-12))
        new_cbs.append({"targets": T, "probs": normalize((1-alpha)*p + alpha*q_mix)})

    return new_cbs, {"enabled": 1.0, "alpha": alpha, "proj": proj, "proj_sigma": proj_sigma}


# ── Per-state statistics ──────────────────────────────────────────────────────

def compute_state_stats_1d(codebooks: List[Dict]) -> List[Dict[str, float]]:
    """Precompute mean (μ) and std (σ) for each smoothed codebook.
    Used by Online Regularization for buffer-drain tilt.
    """
    out = []
    for cb in codebooks:
        T = np.asarray(cb["targets"], np.float64)
        x = T[:, 0] if T.ndim == 2 and T.shape[1] == 1 else T.reshape(-1)
        p = normalize(cb["probs"])
        mu  = float(np.dot(p, x))
        std = float(np.sqrt(max(np.dot(p, (x-mu)**2), 1e-9)))
        out.append({"mu": mu, "std": std})
    return out


# ── Top-level function ────────────────────────────────────────────────────────

def run_template_smoothing(
    stl_config: dict,
    alpha:      float,
    proj:       str   = "soft",
    proj_sigma: float = 1.0,
) -> dict:
    """Apply Template Smoothing to the output of State-Template Learning.

    Parameters
    ----------
    stl_config : dict returned by state_template_learning.run_state_template_learning
    alpha      : smoothing strength α ∈ [0, 1]  (0 = no smoothing)
    proj       : projection method (``"soft"`` recommended)
    proj_sigma : Gaussian kernel bandwidth σ

    Returns
    -------
    smoothed_config : dict
        All STL fields are preserved; ``codebooks_up/down`` are replaced with
        smoothed versions; ``state_stats_up/down`` and ``smoothing`` are added.
    """
    config = dict(stl_config)

    # Upload
    cbs_up, sm_up = smooth_state_codebooks(
        config["codebooks_up"], config["global_codebook_up"],
        alpha=alpha, proj=proj, proj_sigma=proj_sigma)
    stats_up = compute_state_stats_1d(cbs_up)
    print(f"[TS] Upload   alpha={alpha}  "
          f"means={[round(s['mu'],2) for s in stats_up]}")

    # Download
    cbs_dn, sm_dn = smooth_state_codebooks(
        config["codebooks_down"], config["global_codebook_down"],
        alpha=alpha, proj=proj, proj_sigma=proj_sigma)
    stats_dn = compute_state_stats_1d(cbs_dn)
    print(f"[TS] Download alpha={alpha}  "
          f"means={[round(s['mu'],2) for s in stats_dn]}")

    config.update({
        "codebooks_up":    cbs_up,
        "codebooks_down":  cbs_dn,
        "state_stats_up":  stats_up,
        "state_stats_down":stats_dn,
        "smoothing": {
            "enabled": 1.0, "alpha": float(alpha),
            "proj": proj, "proj_sigma": float(proj_sigma),
            "up": sm_up, "down": sm_dn,
        },
    })
    return config


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Template Smoothing")
    p.add_argument("--config", required=True,
                   help="stl_config.pkl produced by state_template_learning.py")
    p.add_argument("--out",    default="./config/smoothed_config.pkl")
    p.add_argument("--alpha",  type=float, default=0.4,
                   help="Smoothing strength α ∈ [0, 1].")
    p.add_argument("--sigma",  type=float, default=1.3,
                   help="Gaussian kernel bandwidth σ.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[TS] Loading STL config from {args.config}")
    with open(args.config, "rb") as f:
        stl_config = pickle.load(f)

    smoothed = run_template_smoothing(stl_config, alpha=args.alpha, proj_sigma=args.sigma)

    with open(args.out, "wb") as f:
        pickle.dump(smoothed, f)
    print(f"[TS] Saved → {args.out}")

    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w") as f:
        json.dump(config_to_jsonable(smoothed), f, indent=2)
    print(f"[TS] JSON  → {json_out}")


if __name__ == "__main__":
    main()
