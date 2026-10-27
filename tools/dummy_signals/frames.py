"""Lazy per-frame values of one (edge, model): object counts, true / false positives, consistency per pair, confidence per frame.

Every value is a pure function of (seed, edge, model, frame): any range can be computed in any order and gives the same numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import poisson

from . import rng

FPS = 6.0


def frame_arrays(w, ei: int, mi: int, f0: int, f1: int) -> dict:
    """Frames f0 .. f1-1 of edge `ei` seen through model `mi`. Pair arrays are aligned with frames f0+1 .. f1-1 (the pair (k-1, k) sits at k)."""
    cfg, seed = w.cfg, w.seed
    F = np.arange(f0, f1, dtype=np.int64)
    t = F / FPS
    a, b, s, label = w.edge_tl[ei].mix(t)
    fr, ar = cfg["frame"], cfg["area_params"]
    # object count: a property of the DATA (edge, frame), never of the model
    lam = (1 - s) * w.lam[a] + s * w.lam[b]
    sg = ar["lambda_slow_sigma"]
    lam = np.maximum(lam * np.exp(sg * rng.value_noise(int(ar["lambda_slow_period_s"] * FPS), F, seed, rng.L_LAMBDA, ei) - sg * sg / 2), 1e-6)
    n_obj = np.minimum(poisson.ppf(rng.uniform(seed, rng.L_NOBJ, ei, F), lam), fr["n_obj_max"]).astype(np.int64)
    if ar["dropout_share"] > 0:                       # stretches without any object (data property): they produce runs of empty frames, as in real footage
        gate = rng.value_noise(int(ar["dropout_period_s"] * FPS), F, seed, rng.L_LAMBDA, ei, 1) > float(ndtri(ar["dropout_share"]))
        n_obj = np.where(gate, n_obj, 0)
    # frame difficulty z: shared by all models of this edge and frame, not across edges
    iid = fr["z_iid_share"]
    z = np.sqrt(iid) * rng.normal(seed, rng.L_Z, ei, F, 0) + np.sqrt(1 - iid) * rng.value_noise(int(fr["z_slow_period_s"] * FPS), F, seed, rng.L_Z, ei, 1)
    Rb = (1 - s) * w.R[mi, a] + s * w.R[mi, b]
    Pb = (1 - s) * w.P[mi, a] + s * w.P[mi, b]
    sh = w.edge_tl[ei].shift(t, mi)                                           # transition effect of a move (0 unless the move has one)
    dP = dR = np.zeros(len(F))
    if sh.any():
        dP, dR = np.clip(Pb + sh, 0.01, 0.995) - Pb, np.clip(Rb + sh, 0.01, 0.995) - Rb
        Pb, Rb = Pb + dP, Rb + dR
    rn, pn = fr["recall_noise"], fr["precision_noise"]
    r = np.clip(Rb + rn["gamma_z"] * z + rn["model_noise"] * rng.normal(seed, rng.L_RECALL, ei, mi, F), 0.01, 0.995)
    p = np.clip(Pb + pn["gamma_z"] * z + pn["model_noise"] * rng.normal(seed, rng.L_PRECISION, ei, mi, F), 0.01, 0.995)
    TP = np.zeros(len(F), dtype=np.int64)
    for i in range(int(n_obj.max()) if len(F) else 0):
        TP += (i < n_obj) & (rng.uniform(seed, rng.L_TP, ei, mi, F, i) < r)
    mu = np.where(TP > 0, TP * (1 - p) / p, fr["fp_floor"])
    FP = poisson.ppf(rng.uniform(seed, rng.L_FP, ei, mi, F), mu).astype(np.int64)
    FN = n_obj - TP
    nbox = TP + FP
    empty = nbox == 0
    # realised per-frame recall / precision minus the cell's values (0 where undefined): the signals also see the quality of the frame itself
    dr = np.nan_to_num(np.where(n_obj > 0, TP / np.maximum(n_obj, 1), np.nan) - Rb)
    dp = np.nan_to_num(np.where(nbox > 0, TP / np.maximum(nbox, 1), np.nan) - Pb)
    # confidence per frame (invalid when the frame has no boxes)
    cf = cfg["confidence_frame"]
    Cfb = (1 - s) * w.Conf[mi, a] + s * w.Conf[mi, b]
    if sh.any():                                                              # the centres follow the shifted quality through their regression on P and R
        Cfb = np.clip(Cfb + cfg["confidence"]["bp"] * dP + cfg["confidence"]["br"] * dR, 0.02, 0.98)
    # the mean confidence of the kept boxes: its noise shrinks with the number of boxes (reference: 8 boxes per frame)
    conf = rng.reflect01(Cfb + cf["gamma_z"] * z + cf["gamma_p_hat"] * dp + cf["gamma_r_hat"] * dr + cf["model_noise"] * np.sqrt(8.0 / np.maximum(nbox, 1)) * rng.normal(seed, rng.L_CONF, ei, mi, F))
    # consistency per pair (k-1, k)
    cs = cfg["consistency_frame"]
    Fp, sp, ap, bp = F[1:], s[1:], a[1:], b[1:]
    Cb = (1 - sp) * w.C[mi, ap] + sp * w.C[mi, bp]
    if sh.any():
        Cb = np.clip(Cb + cfg["consistency"]["bp"] * dP[1:] + cfg["consistency"]["br"] * dR[1:], 0.02, 0.98)
    wstd = cs["wander_calibrated_std"] if cs["wander"] == "calibrated" else float(cs["wander"])
    per = int(cs["wander_period_s"] * FPS)
    sh = cs["wander_shared_share"]
    wander = wstd * (np.sqrt(sh) * rng.value_noise(per, Fp, seed, rng.L_WANDER, ei, 0) + np.sqrt(1 - sh) * rng.value_noise(per, Fp, seed, rng.L_WANDER, ei, mi + 1))
    zbar = (z[:-1] + z[1:]) / 2
    cons = rng.reflect01(Cb + cs["gamma_z"] * zbar + cs["gamma_p_hat"] * (dp[:-1] + dp[1:]) / 2 + cs["gamma_r_hat"] * (dr[:-1] + dr[1:]) / 2 + cs["model_noise"] * rng.normal(seed, rng.L_CONS, ei, mi, Fp) + wander)
    e0, e1 = empty[:-1], empty[1:]
    status = np.where(e0 & e1, 2, np.where(e0 ^ e1, 1, 0))                 # 0 matched, 1 one-sided, 2 empty pair
    cons = np.where(status == 1, 0.0, cons)                                  # real code: exactly one empty frame is a VALID consistency of 0
    cons_valid = status != 2
    cons = np.where(cons_valid, cons, np.nan)
    f1_cell = 2 * Pb * Rb / np.maximum(Pb + Rb, 1e-9)                         # F1 recomputed from the interpolated P and R (not interpolated itself)
    return {"f1_cell": f1_cell, "frame": F, "t": t, "area": label, "n_obj": n_obj, "TP": TP, "FP": FP, "FN": FN, "nbox": nbox, "conf": np.where(empty, np.nan, conf), "conf_valid": ~empty,
            "pair_frame": Fp, "cons": cons, "cons_valid": cons_valid, "cons_status": status}
