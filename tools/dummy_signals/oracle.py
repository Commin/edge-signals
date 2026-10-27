"""Performance oracle: how EVERY model of the world performs on each edge's data, window by window.

This is ground truth about the generated data, not a decision: no model is selected and nothing is switched. For every edge and every window (30 frame pairs = 5 s by default,
the windows of the real engine: window k ends at 5 (k + 1) s) and every model of the world, whatever node stores it, it gives the true F1 (from the TP / FP / FN counts of the
window's frames), the clip consistency and confidence medians (the values `World.clip_score` returns for that window), and where the model is stored. The top 3 of a window are the three
models with the highest true F1 on that edge's data in that window (top1 = the oracle best); best_local is the best model stored on that edge itself (the option that needs no transfer).
The oracle is the unconstrained ideal: every model anywhere, at no cost, no delay and no failure.
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import frames

FPS = frames.FPS
TOP = ("top1", "top2", "top3")
COLUMNS = ["edge", "window", "t_start", "t_end", "area", "model", "size", "version", "location", "location_kind", "stored_on", "true_f1", "consistency", "confidence", "is_initial", "is_top1",
           "initial_model", "initial_f1", "gap",
           "top1_model", "top1_location", "top1_f1", "top2_model", "top2_location", "top2_f1", "top2_delta", "top3_model", "top3_location", "top3_f1", "top3_delta", "top3_summary",
           "best_local_model", "best_local_f1", "best_local_delta"]
SUMMARY_COLUMNS = ["edge", "segment", "kind", "area", "t0", "t1", "n_windows", "initial_model", "initial_mean_f1", "gap",
                   "top1_model", "top1_location", "top1_mean_f1", "top2_model", "top2_location", "top2_mean_f1", "top2_delta", "top3_model", "top3_location", "top3_mean_f1", "top3_delta", "top3_summary",
                   "best_local_model", "best_local_mean_f1", "best_local_delta", "share_consistency_first", "share_confidence_first", "share_both_first"]


def window_pairs(window) -> int:
    """'5s' | '10s' | '1m' | seconds -> number of frame pairs per window (must be a whole number)."""
    sec = cfgmod.parse_duration(window)
    p = sec * FPS
    if sec <= 0 or abs(p - round(p)) > 1e-9:
        raise cfgmod.ConfigError(f"window {window!r}: must be a positive multiple of 1/{int(FPS)} s ({int(FPS)} pairs per second)")
    return int(round(p))


def _nanmedian(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=1)


def _rank(x):
    """Indices of the rows of x (models x windows) by decreasing value in every column (NaN last, stable)."""
    return np.argsort(-np.where(np.isnan(x), -np.inf, x), axis=0, kind="stable")


def top_fields(prefix, ids, loc, order_col, val, k_max=3, mean=False):
    """top1 / top2 / top3 model, location, value and delta to top1 (+ the summary string 'x_a 0.899 | s_a -0.04 | x_b -0.12') from one ranking (order_col: model indices best first; val: values per model)."""
    r = {}
    top1 = float(val[order_col[0]])
    parts = []
    for k in range(k_max):
        key = TOP[k]
        if k < len(order_col):
            mi = int(order_col[k])
            r[f"{key}_model"], r[f"{key}_location"], r[f"{key}_{'mean_f1' if mean else 'f1'}"] = ids[mi], loc[mi][0], float(val[mi])
            if k:
                r[f"{key}_delta"] = float(val[mi]) - top1
            parts.append(f"{ids[mi]} {val[mi]:.3f}" if k == 0 else f"{ids[mi]} {val[mi] - top1:.2f}")
        else:
            r[f"{key}_model"], r[f"{key}_location"], r[f"{key}_{'mean_f1' if mean else 'f1'}"] = "", "", float("nan")
            if k:
                r[f"{key}_delta"] = float("nan")
    r["top3_summary"] = " | ".join(parts)
    return r


def _location(w, e, m):
    """-> (location, location_kind, stored_on): 'local' | id of the other edge / cloud node it is fetched from."""
    src, node = w.model_source(e, m)
    stored = w.model_locations(m)
    if src == "local":
        return "local", "local", ";".join(stored)
    return (node, w.node_index[node]["kind"], ";".join(stored)) if node is not None else ("", "none", "")


def compute(w, window="5s"):
    """Per edge: arrays (models x windows) of true F1, consistency median, confidence median, plus the oracle best of every window."""
    P = window_pairs(window)
    N = w.n_frames
    nwin = (N - 1) // P
    if nwin < 1:
        raise cfgmod.ConfigError("the duration is shorter than one window")
    ids = [m["id"] for m in w.cfg["models"]]
    k = np.arange(nwin)
    out = {"window_pairs": P, "n_windows": nwin, "models": ids, "t_start": (k * P + 1) / FPS, "t_end": (k * P + P) / FPS, "edges": {}}
    for ei, e in enumerate(w.edges):
        f1, cons, conf = (np.full((len(ids), nwin), np.nan) for _ in range(3))
        for mi in range(len(ids)):
            a = frames.frame_arrays(w, ei, mi, 0, N)
            s = slice(1, nwin * P + 1)                                                    # window k scores the frames kP + 1 .. kP + P (the second frame of each of its pairs)
            tp, fp, fn = (a[x][s].reshape(nwin, P).sum(axis=1).astype(float) for x in ("TP", "FP", "FN"))
            den = 2 * tp + fp + fn
            f1[mi] = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), np.nan)
            cons[mi] = _nanmedian(a["cons"][:nwin * P].reshape(nwin, P))                  # pair p is (p - 1, p) and sits at index p - 1: the same 30 pairs as World.clip_score
            conf[mi] = _nanmedian(a["conf"][s].reshape(nwin, P))
        t_mid = (out["t_start"] + out["t_end"]) / 2
        order = _rank(f1)                                                                 # models by decreasing true F1 per window; ties: the first model of the configuration
        local = np.array([m in e["models"] for m in ids])
        best_local = np.argmax(np.where(local[:, None] & ~np.isnan(f1), f1, -np.inf), axis=0)
        out["edges"][e["id"]] = {"area": np.asarray(w.area(e["id"], t_mid)), "f1": f1, "consistency": cons, "confidence": conf, "best": order[0], "order": order, "best_local": best_local,
                                 "initial": ids.index(e["initial_model"]), "location": [_location(w, e["id"], m) for m in ids]}
    return out


def _fmt(x):
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:.6f}"
    return x


def rows(w, o):
    """oracle.csv rows: one per edge x window x model; the window-level columns (top 3, best_local) are repeated on every row of the window."""
    cfgm = {m["id"]: m for m in w.cfg["models"]}
    ids = o["models"]
    for e in w.edges:
        d = o["edges"][e["id"]]
        for j in range(o["n_windows"]):
            f1c = d["f1"][:, j]
            top = top_fields("top", ids, d["location"], d["order"][:, j], f1c)
            bl = int(d["best_local"][j])
            top1 = float(f1c[d["order"][0, j]])
            f1i = f1c[d["initial"]]
            for mi, m in enumerate(ids):
                loc, kind, stored = d["location"][mi]
                yield {"edge": e["id"], "window": j, "t_start": o["t_start"][j], "t_end": o["t_end"][j], "area": d["area"][j], "model": m, "size": cfgm[m]["size"], "version": cfgm[m]["version"],
                       "location": loc, "location_kind": kind, "stored_on": stored, "true_f1": f1c[mi], "consistency": d["consistency"][mi, j], "confidence": d["confidence"][mi, j],
                       "is_initial": mi == d["initial"], "is_top1": mi == int(d["order"][0, j]), "initial_model": ids[d["initial"]], "initial_f1": f1i, "gap": top1 - f1i, **top,
                       "best_local_model": ids[bl], "best_local_f1": f1c[bl], "best_local_delta": f1c[bl] - top1}


def _first(score, best):
    """Share of windows in which the model with the highest score is the window's oracle best (a window without any valid score does not count)."""
    s = np.where(np.isnan(score), -np.inf, score)
    return float(np.mean(np.argmax(s, axis=0) == best))


def summary_rows(w, o):
    """oracle_summary.csv rows: one per edge x route segment (a window belongs to the segment that contains its middle); top 3 and best_local are ranked by the segment-mean F1."""
    ids = o["models"]
    t_mid = (o["t_start"] + o["t_end"]) / 2
    for e in w.edges:
        d = o["edges"][e["id"]]
        local = np.array([m in e["models"] for m in ids])
        for si, seg in enumerate(w.segments(e["id"])):
            sel = (t_mid >= seg["t0"]) & (t_mid < seg["t1"])
            if not sel.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(d["f1"][:, sel], axis=1)
            order = _rank(mean[:, None])[:, 0]
            top = top_fields("top", ids, d["location"], order, mean, mean=True)
            bl = int(np.argmax(np.where(local & ~np.isnan(mean), mean, -np.inf)))
            kind = seg.get("kind", "stay")
            cons, conf = d["consistency"][:, sel], d["confidence"][:, sel]
            both = np.where(np.isnan(cons) | np.isnan(conf), np.nan, (cons + conf) / 2)
            best = d["best"][sel]
            yield {"edge": e["id"], "segment": si, "kind": kind, "area": seg["area"] if kind == "stay" else f"{seg['from_area']}->{seg['area']}", "t0": seg["t0"], "t1": seg["t1"],
                   "n_windows": int(sel.sum()), "initial_model": ids[d["initial"]], "initial_mean_f1": mean[d["initial"]], "gap": mean[order[0]] - mean[d["initial"]], **top,
                   "best_local_model": ids[bl], "best_local_mean_f1": mean[bl], "best_local_delta": mean[bl] - mean[order[0]],
                   "share_consistency_first": _first(cons, best), "share_confidence_first": _first(conf, best), "share_both_first": _first(both, best)}


def write(w, out_dir, window="5s"):
    """Writes oracle.csv, oracle_summary.csv and world.json into out_dir (nothing else). Returns the computed oracle."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = compute(w, window)
    w.save(out / "world.json")
    for name, cols, gen in (("oracle.csv", COLUMNS, rows(w, o)), ("oracle_summary.csv", SUMMARY_COLUMNS, summary_rows(w, o))):
        with open(out / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for r in gen:
                wr.writerow({c: _fmt(r.get(c, "")) for c in cols})
    return o
