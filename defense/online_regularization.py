#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
online_regularization.py — Module 3: Online Regularization
===========================================================
Applies the learned shaping policy to network traces at inference time.

Pipeline position
-----------------
  smoothed_config.pkl  +  data splits  →  [Online Regularization]  →  defended/

How it works
------------
For each time slot *t* in a trace:

1. **State estimation** – compute the K-slot sliding-window arrival count
   Λ_t for upload and download independently; map Λ_t to a traffic state via
   the thresholds learned by State-Template Learning.

2. **Template selection** – draw a per-slot sending budget from the smoothed
   codebook of the current state.  When a packet backlog exists the
   distribution is tilted toward budgets that drain the queue within *X*
   slots (buffer draining, controlled by ``tradeoff``).

3. **Intra-slot scheduling** – assign each packet a timestamp drawn from a
   Rayleigh distribution within the slot; pad remaining slots with dummy
   packets up to the target budget.

4. **Tail padding** – continue emitting dummy traffic for
   ``idle_slots_no_real`` slots after the last real packet arrives.

Input
-----
* ``smoothed_config.pkl`` — output of template_smoothing.py.
* Dataset directory containing ``train.npz``, ``valid.npz``, ``test.npz``.

Output
------
* ``{out_dir}/{split}.npz`` — defended traces in WFLib NPZ format.
* ``{out_dir}/overhead.json`` — bandwidth overhead (BOH) and time overhead (TOH).
* ``{out_dir}/config.json`` — full runtime config snapshot.

Usage
-----
::

    python online_regularization.py \\
        --config   ./config/smoothed_config.pkl \\
        --data_dir ./data/closed_world \\
        --out_dir  ./defended \\
        --tradeoff 0.05 --drain_X 4 --idle_slots_no_real 2
"""

import argparse
import json
import os
import pickle
import shutil
import time
from collections import deque
from multiprocessing import Process
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from utils import load_npz_timeseries, normalize, config_to_jsonable, EPS_TIME_SHIFT

# Global variables shared across worker processes (written before forking)
GLOBAL_times  = None
GLOBAL_dirs   = None
GLOBAL_labels = None
GLOBAL_config = None


# ── Online helpers ────────────────────────────────────────────────────────────

def map_lambda_to_state(Lambda_t: float, thresholds: List[float]) -> int:
    s = 0
    for thr in thresholds:
        if Lambda_t >= thr:
            s += 1
    return s


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    ex = np.exp(logits)
    s  = float(ex.sum())
    return ex / s if s > 0 else np.ones_like(ex) / len(ex)


def sample_rayleigh_in_slot(
    slot_start: float,
    delta:      float,
    count:      int,
    scale:      Optional[float] = None,
    eps:        float = 1e-9,
) -> np.ndarray:
    """Sample *count* packet timestamps within [slot_start, slot_start + delta)."""
    if count <= 0:
        return np.empty(0, np.float64)
    if scale is None:
        scale = float(delta) / 3.0
    ts = float(slot_start) + np.random.rayleigh(float(scale), int(count))
    return np.clip(ts, float(slot_start),
                   float(slot_start) + float(delta) - float(eps)).astype(np.float64)


def sample_template_count_1d(
    codebook:   Dict[str, np.ndarray],
    state_stat: Dict[str, float],
    backlog:    int,
    tradeoff:   float,
    drain_X:    int = 5,
) -> int:
    """Sample the per-slot sending budget for one direction.

    The template distribution is optionally tilted toward budgets that can
    drain a packet backlog within *drain_X* slots, controlled by *tradeoff*.
    """
    backlog  = int(max(backlog, 0))
    X        = int(max(drain_X, 1))
    tradeoff = float(np.clip(tradeoff, 0.0, 1.0))

    T     = np.asarray(codebook["targets"], np.int32)
    means = (T[:, 0] if T.ndim == 2 and T.shape[1] == 1
             else T.reshape(-1)).astype(np.float64)
    base_p = normalize(np.asarray(codebook["probs"], np.float64))

    if means.size <= 1:
        mean = float(means[0]) if means.size else 0.0
    else:
        mu  = float(state_stat.get("mu",  float(np.mean(means))))
        std = float(max(state_stat.get("std", float(np.std(means) + 1e-6)), 1e-6))
        gap = float(backlog) - X * mu

        if gap <= 0.0 or tradeoff <= 0.0:
            p = base_p
        else:
            need     = float(backlog) / X
            strength = tradeoff * float(max(gap / (X * std + 1.0), 0.0))
            diff     = means - need
            score    = -np.abs(diff) - np.maximum(-diff, 0.0)
            logits   = np.log(base_p + 1e-15) + strength * (score / (std + 1e-6))
            p        = _softmax(logits)

        mean = float(means[int(np.random.choice(len(means), p=p))])

    mean_i = int(max(round(mean), 0))
    if mean_i == 0:
        return 0

    import math
    c   = int(np.random.poisson(mean_i))
    cap = int(mean_i + 3.0 * math.sqrt(mean_i + 1.0) + 2.0)
    return int(min(max(c, 0), cap))


# ── Overhead computation ──────────────────────────────────────────────────────

def compute_overhead_aligned(
    orig_time:  np.ndarray,
    def_time:   np.ndarray,
    dummy_mask: np.ndarray,
) -> Tuple[float, float, int, float, int, float]:
    """Return (BOH, TOH, extra_pkts, extra_time, n_real, t_last_orig)."""
    real_mask   = orig_time > 0
    n_real      = int(real_mask.sum())
    if n_real == 0:
        return 0.0, 0.0, 0, 0.0, 0, 0.0
    t_last_orig = float(orig_time[real_mask][-1])
    if def_time.size == 0:
        return 0.0, 0.0, 0, 0.0, n_real, t_last_orig

    n_dummy    = int(dummy_mask.sum())
    BOH        = n_dummy / n_real
    real_def   = ~dummy_mask
    t_star     = float(def_time[real_def][-1]) if real_def.any() else float(def_time[-1])
    TOH        = t_star / t_last_orig - 1.0 if t_last_orig > 0 else 0.0
    extra_pkts = int(def_time.size) - n_real
    extra_time = float(def_time[-1]) - t_last_orig
    return BOH, TOH, extra_pkts, extra_time, n_real, t_last_orig


# ── Per-process worker ────────────────────────────────────────────────────────

def _defend_range(cur_range, idx, split_name, tmp_dir):
    """Defend a slice of traces (runs in a child process)."""
    global GLOBAL_times, GLOBAL_dirs, GLOBAL_labels, GLOBAL_config
    times, dirs, labels, config = (GLOBAL_times, GLOBAL_dirs,
                                   GLOBAL_labels, GLOBAL_config)
    start, end = cur_range

    delta              = float(config["delta"])
    K_slot             = int(config["K_slot"])
    thresholds_up      = config.get("thresholds_up",    [])
    thresholds_down    = config.get("thresholds_down",  [])
    num_states_up      = int(config.get("num_states_up",   1))
    num_states_down    = int(config.get("num_states_down", 1))
    codebooks_up       = config["codebooks_up"]
    codebooks_down     = config["codebooks_down"]
    stats_up           = config.get("state_stats_up",   [{} for _ in range(num_states_up)])
    stats_down         = config.get("state_stats_down", [{} for _ in range(num_states_down)])
    tradeoff           = float(config.get("tradeoff",            0.5))
    idle_slots_no_real = int(config.get("idle_slots_no_real",    2))
    rayleigh_ratio     = float(config.get("rayleigh_scale_ratio",0.3))
    eps                = float(config.get("eps_time_shift",       EPS_TIME_SHIFT))
    drain_X            = int(config.get("drain_X",               5))

    N, L    = times.shape
    n_local = end - start
    new_X   = np.zeros((n_local, L), np.float64)
    y_local = labels[start:end].copy()

    Bs, Ts, ep_l, et_l, nr_l, tl_l = [], [], [], [], [], []
    np.random.seed(int(time.time()) + idx * 13337)

    for local_i, i in enumerate(range(start, end)):
        t_seq, d_seq = times[i], dirs[i]
        mask = t_seq != 0
        if not np.any(mask):
            for lst in (Bs, Ts, ep_l, et_l, nr_l, tl_l):
                lst.append(0)
            continue

        t_nz    = t_seq[mask];  d_nz = d_seq[mask]
        start_t = float(t_nz[0])
        sidx    = np.maximum(
            np.floor((t_nz - start_t) / delta).astype(np.int64), 0)

        q_up, q_dn  = deque(), deque()
        sched_times = np.zeros(len(t_nz), np.float64)
        dum_t: List[float] = [];  dum_d: List[int] = []

        p = 0;  slot_t = 0.0;  s = 0
        idle_up = idle_dn = 0
        lam_up  = lam_dn  = 0.0

        K_eff = int(max(K_slot, 0))
        if K_eff:
            h_up = deque([], maxlen=K_eff);  s_up = 0.0
            h_dn = deque([], maxlen=K_eff);  s_dn = 0.0
        else:
            h_up = h_dn = None;  s_up = s_dn = 0.0

        while True:
            if (p >= len(t_nz)
                    and not q_up and not q_dn
                    and idle_up >= idle_slots_no_real
                    and idle_dn >= idle_slots_no_real):
                break

            # Fast-forward empty gaps
            if (not q_up and not q_dn
                    and idle_up >= idle_slots_no_real
                    and idle_dn >= idle_slots_no_real
                    and p < len(t_nz)):
                ns = int(sidx[p])
                if ns > s:
                    slot_t += delta * (ns - s);  s = ns;  continue

            arr_up = arr_dn = 0
            while p < len(t_nz) and int(sidx[p]) <= s:
                if d_nz[p] > 0:   q_up.append(p); arr_up += 1
                elif d_nz[p] < 0: q_dn.append(p); arr_dn += 1
                p += 1

            bl_up = len(q_up);  bl_dn = len(q_dn)

            # Tail-padding decision per direction
            def _pad(bl, arr, idle):
                if bl == 0 and arr == 0:
                    return (True, idle+1) if idle < idle_slots_no_real else (False, idle)
                return True, 0

            pad_up, idle_up = _pad(bl_up, arr_up, idle_up)
            pad_dn, idle_dn = _pad(bl_dn, arr_dn, idle_dn)

            def _tgt(pad, bl, lam, thrs, n_st, cbs, sts):
                if not pad:
                    return 0
                st = int(np.clip(map_lambda_to_state(lam + bl, thrs), 0, n_st-1))
                return sample_template_count_1d(
                    cbs[st], sts[st] if st < len(sts) else {},
                    bl, tradeoff, drain_X)

            tgt_up = _tgt(pad_up, bl_up, lam_up, thresholds_up,
                          num_states_up, codebooks_up, stats_up)
            tgt_dn = _tgt(pad_dn, bl_dn, lam_dn, thresholds_down,
                          num_states_down, codebooks_down, stats_down)

            def _sched(q, target, d_dir):
                real_n = min(len(q), target) if q else 0
                if real_n:
                    ts = sample_rayleigh_in_slot(
                        slot_t, delta, real_n, delta * rayleigh_ratio)
                    for j in range(real_n):
                        sched_times[q.popleft()] = float(ts[j])
                dummy_n = max(0, target - real_n)
                if dummy_n:
                    ts = sample_rayleigh_in_slot(
                        slot_t, delta, dummy_n, delta * rayleigh_ratio)
                    for t in ts:
                        dum_t.append(float(t)); dum_d.append(d_dir)

            _sched(q_up, tgt_up, +1)
            _sched(q_dn, tgt_dn, -1)

            if K_eff:
                if len(h_up) == K_eff: s_up -= float(h_up[0])
                h_up.append(float(arr_up)); s_up += float(arr_up); lam_up = s_up
                if len(h_dn) == K_eff: s_dn -= float(h_dn[0])
                h_dn.append(float(arr_dn)); s_dn += float(arr_dn); lam_dn = s_dn

            slot_t += delta;  s += 1

        # Assemble defended trace
        rt = sched_times + start_t
        rd = d_nz.astype(np.int8)
        if dum_t:
            dt    = np.asarray(dum_t, np.float64) + start_t
            dd    = np.asarray(dum_d, np.int8)
            all_t = np.concatenate([rt, dt])
            all_d = np.concatenate([rd, dd])
            dmask = np.zeros(len(all_t), bool)
            dmask[len(rt):] = True
        else:
            all_t = rt;  all_d = rd;  dmask = np.zeros(len(rt), bool)

        order = np.argsort(all_t, kind="mergesort")
        all_t = all_t[order] + eps
        all_d = all_d[order]
        dmask = dmask[order]

        BOH, TOH, ep, et, nr, tlo = compute_overhead_aligned(t_seq, all_t, dmask)
        Bs.append(BOH); Ts.append(TOH)
        ep_l.append(ep); et_l.append(et)
        nr_l.append(nr); tl_l.append(tlo)

        keep = min(len(all_t), L)
        if keep:
            new_X[local_i, :keep] = all_d[:keep].astype(np.float64) * all_t[:keep]

    np.savez(os.path.join(tmp_dir, f"{split_name}_part_{idx}.npz"),
             X=new_X, y=y_local,
             B=np.array(Bs, np.float64),    T=np.array(Ts, np.float64),
             extra_pkts=np.array(ep_l, np.int64), extra_time=np.array(et_l, np.float64),
             n_real=np.array(nr_l, np.int64),      t_last_orig=np.array(tl_l, np.float64))


# ── Multi-process driver ──────────────────────────────────────────────────────

def apply_defense_to_split(
    times:      np.ndarray,
    dirs:       np.ndarray,
    labels:     np.ndarray,
    config:     dict,
    split_name: str,
    out_dir:    str,
    workers:    int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Defend all traces in one dataset split using *workers* parallel processes."""
    global GLOBAL_times, GLOBAL_dirs, GLOBAL_labels, GLOBAL_config
    GLOBAL_times  = times;   GLOBAL_dirs   = dirs
    GLOBAL_labels = labels;  GLOBAL_config = config

    N, L = times.shape
    tmp  = os.path.join(out_dir, f"tmp_{split_name}")
    os.makedirs(tmp, exist_ok=True)
    print(f"[OR] {split_name}: N={N}, L={L}")

    m   = max(min(workers, N), 1)
    ns, ex = divmod(N, m)
    cuts   = np.array([0] + ex*[ns+1] + (m-ex)*[ns]).cumsum()
    rngs   = list(zip(cuts, cuts[1:]))

    pool = [Process(target=_defend_range, args=(r, i, split_name, tmp))
            for i, r in enumerate(rngs)]
    for proc in pool: proc.start()
    for proc in pool: proc.join()

    Xl, yl = [], []
    Bl, Tl, epl, etl, nrl, tll = [], [], [], [], [], []
    for i in range(len(rngs)):
        pt = np.load(os.path.join(tmp, f"{split_name}_part_{i}.npz"))
        Xl.append(pt["X"]);            yl.append(pt["y"])
        Bl.append(pt["B"]);            Tl.append(pt["T"])
        epl.append(pt["extra_pkts"]); etl.append(pt["extra_time"])
        nrl.append(pt["n_real"]);     tll.append(pt["t_last_orig"])
    shutil.rmtree(tmp, ignore_errors=True)

    X_def = np.concatenate(Xl) if Xl else np.zeros((0, L))
    y_def = np.concatenate(yl) if yl else np.zeros(0)
    ep    = np.concatenate(epl); et  = np.concatenate(etl)
    nr    = np.concatenate(nrl); tlo = np.concatenate(tll)
    Bs    = np.concatenate(Bl);  Ts  = np.concatenate(Tl)

    tot_ep = int(ep.sum()); tot_et = float(et.sum())
    tot_nr = int(nr.sum()); tot_tlo= float(tlo.sum())
    overhead = {
        "n_flows":                  int(X_def.shape[0]),
        "bandwidth_overhead":       float(Bs.mean()) if Bs.size else 0.0,
        "time_overhead":            float(Ts.mean()) if Ts.size else 0.0,
        "bandwidth_overhead_total": tot_ep / tot_nr  if tot_nr  else 0.0,
        "time_overhead_total":      tot_et / tot_tlo if tot_tlo else 0.0,
        "total_extra_pkts":  tot_ep,  "total_extra_time": tot_et,
        "total_real_pkts":   tot_nr,  "total_real_time":  tot_tlo,
    }
    return X_def, y_def, overhead


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Online Regularization")
    p.add_argument("--config",               required=True,
                   help="smoothed_config.pkl produced by template_smoothing.py")
    p.add_argument("--data_dir",             required=True)
    p.add_argument("--out_dir",              default="./defended")
    p.add_argument("--workers",              type=int,   default=32)
    p.add_argument("--tradeoff",             type=float, default=0.05)
    p.add_argument("--idle_slots_no_real",   type=int,   default=2)
    p.add_argument("--rayleigh_scale_ratio", type=float, default=0.15)
    p.add_argument("--drain_X",              type=int,   default=4)
    p.add_argument("--eps_time_shift",       type=float, default=EPS_TIME_SHIFT)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[OR] Loading config from {args.config}")
    with open(args.config, "rb") as f:
        config = pickle.load(f)

    config.update({
        "tradeoff":            float(np.clip(args.tradeoff, 0.0, 1.0)),
        "idle_slots_no_real":  int(args.idle_slots_no_real),
        "rayleigh_scale_ratio":float(args.rayleigh_scale_ratio),
        "drain_X":             int(max(args.drain_X, 1)),
        "eps_time_shift":      float(args.eps_time_shift),
    })

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config_to_jsonable(config), f, indent=2)

    overhead: dict = {}
    splits_done: List[str] = []

    for split in ["train", "valid", "test"]:
        path = os.path.join(args.data_dir, f"{split}.npz")
        if not os.path.isfile(path):
            print(f"[OR] Skipping {split}: not found")
            continue
        t, d, lbl = load_npz_timeseries(path)
        X_def, y_def, oh = apply_defense_to_split(
            t, d, lbl, config, split, args.out_dir, args.workers)
        np.savez(os.path.join(args.out_dir, f"{split}.npz"), X=X_def, y=y_def)
        print(f"[OR] {split}: BOH={oh['bandwidth_overhead']:.2%}  "
              f"TOH={oh['time_overhead']:.2%}")
        overhead[split] = oh
        splits_done.append(split)

    if splits_done:
        ep_t  = sum(overhead[s]["total_extra_pkts"] for s in splits_done)
        et_t  = sum(overhead[s]["total_extra_time"] for s in splits_done)
        nr_t  = sum(overhead[s]["total_real_pkts"]  for s in splits_done)
        tlo_t = sum(overhead[s]["total_real_time"]  for s in splits_done)
        overhead["overall"] = {
            "bandwidth_overhead":       float(np.mean([overhead[s]["bandwidth_overhead"] for s in splits_done])),
            "time_overhead":            float(np.mean([overhead[s]["time_overhead"]      for s in splits_done])),
            "bandwidth_overhead_total": ep_t  / nr_t  if nr_t  else 0.0,
            "time_overhead_total":      et_t  / tlo_t if tlo_t else 0.0,
            "n_flows":                  sum(overhead[s]["n_flows"] for s in splits_done),
            "total_extra_pkts": ep_t,  "total_extra_time": et_t,
            "total_real_pkts":  nr_t,  "total_real_time":  tlo_t,
        }
        ov = overhead["overall"]
        print(f"\n[OR] Overall: BOH={ov['bandwidth_overhead']:.2%}  "
              f"TOH={ov['time_overhead']:.2%}")

    with open(os.path.join(args.out_dir, "overhead.json"), "w") as f:
        json.dump(overhead, f, indent=2)
    print(f"[OR] Done. Results → {args.out_dir}/")


if __name__ == "__main__":
    main()
