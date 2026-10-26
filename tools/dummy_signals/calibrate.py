#!/usr/bin/env python3
"""Pair-level calibration of the per-frame couplings: a documented, reproducible search.

Calibration world: ONE model, ONE edge, 4 areas in the proportions 0.52 / 0.34 / 0.08 / 0.06 whose F1 is 0.94 / 0.89 / 0.57 / 0.86 (a high-quality area,
a second healthy one, a hard low-quality one and a hazy one); 6,300 frame transitions per seed. On every transition (t, t+1):
  recall event    = recall(t) < 0.7 or recall(t+1) < 0.7        precision event = the same for precision
  recall = TP / n_obj (undefined when n_obj = 0), precision = TP / (TP + FP) (undefined when TP + FP = 0); a transition is valid for a label only if both frames are defined.
  consistency alarm = 1 - consistency of the pair; confidence score = - (mean confidence of frame t+1).
The targets below (scale-free statistics of real per-frame detections) are matched in the mean over seeds. The absolute levels of consistency / confidence stay anchored to the
real healthy footage and enter the search as constraints. Nelder-Mead from the defaults, seeds fixed (0..19); run `calibrate.py check` on other seeds.

    python tools/dummy_signals/calibrate.py fit     # prints the best parameters and the achieved statistics
    python tools/dummy_signals/calibrate.py check   # the statistics of the current default_config.yaml on seeds 100..119
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.dummy_signals import frames  # noqa: E402
from tools.dummy_signals.config import deep_merge  # noqa: E402
from tools.dummy_signals.world import World  # noqa: E402

F1_AREAS = [0.94, 0.89, 0.57, 0.86]
PROPORTIONS = [0.52, 0.34, 0.08, 0.06]
N_TRANSITIONS = 6300
SEEDS_FIT = list(range(20))
SEEDS_CHECK = list(range(100, 120))

# name: (target, tolerance used by the search and by the verification)
TARGETS = {
    "event_rate_recall": (0.091, 0.03), "event_rate_precision": (0.126, 0.03),
    "auroc_consistency_recall": (0.80, 0.05), "auroc_consistency_precision": (0.84, 0.05),
    "auroc_confidence_recall": (0.79, 0.05), "auroc_confidence_precision": (0.86, 0.05),
    "spearman_consistency_min_recall": (0.35, 0.10), "spearman_consistency_min_precision": (0.56, 0.10),
    "spearman_confidence_min_recall": (0.37, 0.10), "spearman_confidence_min_precision": (0.59, 0.10),
    "spearman_consistency_confidence": (0.63, 0.10),
    "zero_share_overall": (0.023, 0.03), "zero_share_hard_area": (0.30, 0.10), "zero_share_high_areas_max": (0.0, 0.01)}
# absolute levels of the healthy area (real healthy footage, log A): constraints of the search
ABSOLUTE = {"consistency_median": (0.788, 0.03), "consistency_p5": (0.656, 0.05), "consistency_p95": (0.912, 0.05), "consistency_within_window_std": (0.066, 0.012),
            "confidence_median": (0.6625, 0.03), "confidence_p5": (0.583, 0.05), "confidence_p95": (0.736, 0.05), "confidence_within_window_std": (0.036, 0.008)}

PARAMS = ["gamma_r", "gamma_p", "model_noise_r", "model_noise_p", "gamma_c", "model_noise_c", "gamma_f", "model_noise_f", "pr_ratio_0", "pr_ratio_1", "pr_ratio_2", "pr_ratio_3", "lambda_hard_area",
          "kappa_c_p", "kappa_c_r", "kappa_f_p", "kappa_f_r"]
START = [0.0347, 0.0802, 0.009, 0.0007, 0.102, 0.0176, 0.0498, 0.0057, 0.993, 0.941, 0.815, 0.906, 3.06, 0.0, 0.0, 0.0, 0.0]          # (the best point of a search without the kappa terms)
LOW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.005, 0.6, 0.6, 0.5, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0]
HIGH = [0.30, 0.30, 0.30, 0.30, 0.25, 0.20, 0.25, 0.20, 1.6, 1.6, 2.0, 1.6, 8.0, 0.6, 0.6, 0.6, 0.6]


def calibration_config(theta: dict, defaults: dict | None = None) -> dict:
    ts = [theta[f"pr_ratio_{k}"] for k in range(4)]                             # P / R of each area's cell (F1 fixed)
    R = [f * (1 + t) / (2 * t) for f, t in zip(F1_AREAS, ts)]
    P = [t * r for t, r in zip(ts, R)]
    n = N_TRANSITIONS
    segs, order = [], [0, 1, 0, 2, 1, 0, 3, 1, 0, 2, 1, 0, 3, 1, 0, 1, 0, 2, 3, 1, 0]       # blocks of each area, interleaved
    counts = {k: order.count(k) for k in range(4)}
    for k in order:
        segs.append([k, n / 6.0 * PROPORTIONS[k] / counts[k]])
    cfg = {"simple": {"duration": f"{n / 6.0}s", "n_areas": 4, "n_edges": 1, "steepness": 1.0}, "edges": [{"id": "e1", "position": 0}], "local_episodes": {"rate_per_hour": 0},
           "timeline": {"fixed_segments": segs}, "models": [{"id": "m", "size": "s", "version": "a", "trained_areas": [0, 1, 2, 3]}],
           "matrix": {"fixed": {"m": {"P": P, "R": R}}}, "blind": {"beta": 0.0},
           "area_params": {"lambda_obj": [9.0, 9.0, theta["lambda_hard_area"], 9.0], "lambda_floor": 0.1, "dropout_share": 0.0},
           "frame": {"recall_noise": {"gamma_z": theta["gamma_r"], "model_noise": theta["model_noise_r"]}, "precision_noise": {"gamma_z": theta["gamma_p"], "model_noise": theta["model_noise_p"]}},
           "consistency_frame": {"gamma_z": theta["gamma_c"], "model_noise": theta["model_noise_c"], "gamma_p_hat": theta["kappa_c_p"], "gamma_r_hat": theta["kappa_c_r"]},
           "confidence_frame": {"gamma_z": theta["gamma_f"], "model_noise": theta["model_noise_f"], "gamma_p_hat": theta["kappa_f_p"], "gamma_r_hat": theta["kappa_f_r"]}}
    return deep_merge(defaults or {}, cfg)


def auroc(score, event):
    event = np.asarray(event, bool)
    n1, n0 = event.sum(), (~event).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_metrics(w) -> dict:
    """The statistics of one world (one seed)."""
    a = frames.frame_arrays(w, 0, 0, 0, w.n_frames + 1)
    n, tp, fp = a["n_obj"], a["TP"], a["FP"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(n > 0, tp / np.maximum(n, 1), np.nan)
        pre = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    cons, cv, conf, cfv = a["cons"], a["cons_valid"], a["conf"][1:], a["conf_valid"][1:]
    reg = a["area"][1:]
    out = {}
    for name, x in (("recall", rec), ("precision", pre)):
        m = np.minimum(x[:-1], x[1:])
        ok = ~np.isnan(m)                                         # both frames defined for this label
        ev = m < 0.7
        out[f"event_rate_{name}"] = float(ev[ok].mean())
        okc = ok & cv
        out[f"auroc_consistency_{name}"] = auroc(1 - cons[okc], ev[okc])
        okf = ok & cfv
        out[f"auroc_confidence_{name}"] = auroc(-conf[okf], ev[okf])
        out[f"spearman_consistency_min_{name}"] = float(spearmanr(cons[okc], m[okc])[0])
        out[f"spearman_confidence_min_{name}"] = float(spearmanr(conf[okf], m[okf])[0])
    both = cv & cfv
    out["spearman_consistency_confidence"] = float(spearmanr(cons[both], conf[both])[0])
    z = (cons == 0) & cv
    out["zero_share_overall"] = float(z.sum() / cv.sum())
    per = {int(k): float(z[(reg == k) & cv].sum() / max(((reg == k) & cv).sum(), 1)) for k in range(4)}
    out["zero_share_by_area"] = per
    out["zero_share_hard_area"] = per[2]
    out["zero_share_high_areas_max"] = max(per[0], per[1], per[3])
    out["empty_pair_share_by_area"] = {int(k): float((a["cons_status"][reg == k] == 2).mean()) for k in range(4)}
    out["one_sided_share_by_area"] = {int(k): float((a["cons_status"][reg == k] == 1).mean()) for k in range(4)}
    out["no_boxes_share_by_area"] = {int(k): float((~a["conf_valid"][1:][reg == k]).mean()) for k in range(4)}
    # absolute levels in the healthy area 0 (interior of its blocks)
    m0 = (reg == 0) & cv
    c0, f0 = cons[m0], conf[(reg == 0) & cfv]
    n30 = (len(cons) // 30) * 30
    blk = np.where(cv, cons, np.nan)[:n30].reshape(-1, 30)
    lab = reg[:n30].reshape(-1, 30)
    pure = (lab == 0).all(axis=1)
    cblk = np.where(cfv, conf, np.nan)[:n30].reshape(-1, 30)
    sd = lambda b: float(np.nanmedian(np.nanstd(b[pure], axis=1))) if pure.any() else np.nan
    out.update({"consistency_median": float(np.median(c0)), "consistency_p5": float(np.percentile(c0, 5)), "consistency_p95": float(np.percentile(c0, 95)), "consistency_within_window_std": sd(blk),
                "confidence_median": float(np.median(f0)), "confidence_p5": float(np.percentile(f0, 5)), "confidence_p95": float(np.percentile(f0, 95)), "confidence_within_window_std": sd(cblk)})
    return out


def evaluate(theta: dict, seeds, defaults=None) -> dict:
    ms = [pair_metrics(World(calibration_config(theta, defaults), s)) for s in seeds]
    keys = [k for k in ms[0] if not isinstance(ms[0][k], dict)]
    res = {k: (float(np.nanmean([m[k] for m in ms])), float(np.nanstd([m[k] for m in ms]))) for k in keys}
    for k in ("zero_share_by_area", "empty_pair_share_by_area", "one_sided_share_by_area", "no_boxes_share_by_area"):
        res[k] = {r: float(np.mean([m[k][r] for m in ms])) for r in ms[0][k]}
    return res


def loss(vec, seeds=SEEDS_FIT, defaults=None, verbose=False):
    v = np.clip(vec, LOW, HIGH)
    r = evaluate(dict(zip(PARAMS, v)), seeds, defaults)
    tot = 0.0
    for k, (tgt, tol) in {**TARGETS, **ABSOLUTE}.items():
        x = r[k][0]
        err = 0.0 if np.isnan(x) else (x - tgt) / tol
        if k == "zero_share_high_areas_max":
            err = max(x - 0.0, 0) / tol
        tot += err ** 2
    return tot


def fit(maxiter=40, defaults=None, workers=16):
    """Differential evolution over the parameter box (global), then Nelder-Mead from its best point (local polish). Seeds fixed."""
    from scipy.optimize import differential_evolution, minimize
    de = differential_evolution(loss, list(zip(LOW, HIGH)), args=(SEEDS_FIT[:10], defaults), maxiter=maxiter, popsize=10, seed=1, workers=workers, updating="deferred", tol=1e-4, polish=False, init="sobol", x0=START)
    res = minimize(loss, de.x, args=(SEEDS_FIT, defaults), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-3, "adaptive": True})
    best = dict(zip(PARAMS, np.clip(res.x, LOW, HIGH)))
    return best, res.fun, de.nit + res.nit


def report(theta, seeds, defaults=None):
    r = evaluate(theta, seeds, defaults)
    rows = {}
    for k, (tgt, tol) in {**TARGETS, **ABSOLUTE}.items():
        m, sd = r[k]
        ok = (m <= tol) if k == "zero_share_high_areas_max" else abs(m - tgt) <= tol
        rows[k] = {"target": tgt, "tolerance": tol, "mean": m, "sd": sd, "ok": bool(ok)}
    return rows, r


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "fit":
        best, f, nit = fit()
        print(json.dumps({"best": best, "loss": f, "iterations": nit}, indent=1))
        rows, _ = report(best, SEEDS_FIT)
    else:
        cur = World({}, 0).cfg
        best = {"gamma_r": cur["frame"]["recall_noise"]["gamma_z"], "gamma_p": cur["frame"]["precision_noise"]["gamma_z"], "model_noise_r": cur["frame"]["recall_noise"]["model_noise"],
                "model_noise_p": cur["frame"]["precision_noise"]["model_noise"], "gamma_c": cur["consistency_frame"]["gamma_z"], "model_noise_c": cur["consistency_frame"]["model_noise"],
                "gamma_f": cur["confidence_frame"]["gamma_z"], "model_noise_f": cur["confidence_frame"]["model_noise"], "kappa_c_p": cur["consistency_frame"]["gamma_p_hat"], "kappa_c_r": cur["consistency_frame"]["gamma_r_hat"],
                "kappa_f_p": cur["confidence_frame"]["gamma_p_hat"], "kappa_f_r": cur["confidence_frame"]["gamma_r_hat"],
                "lambda_hard_area": cur["calibration"]["lambda_hard_area"], **{f"pr_ratio_{k}": r for k, r in enumerate(cur["calibration"]["pr_ratio"])}}
        rows, _ = report(best, SEEDS_CHECK)
    for k, v in rows.items():
        print(f"{'OK ' if v['ok'] else 'OUT'} {k:42s} target {v['target']:.3f} +-{v['tolerance']:.3f}  mean {v['mean']:.3f} sd {v['sd']:.3f}")
