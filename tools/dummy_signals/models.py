"""Model x area matrices: precision P, recall R, F1, consistency centre C, confidence centre Conf, and the blind cells."""
from __future__ import annotations

import numpy as np

from . import config as cfgmod
from . import rng


def f1_of(P, R):
    return 2 * P * R / np.maximum(P + R, 1e-9)


def anchors(cfg):
    """Intercepts from the anchor: the healthy cell (P = R = anchor F1) has the real healthy level of log A."""
    an, c, f = cfg["anchor"], cfg["consistency"], cfg["confidence"]
    return (an["consistency"] - (c["bp"] + c["br"]) * an["f1"], an["confidence"] - (f["bp"] + f["br"]) * an["f1"])


def build(cfg, seed, difficulty):
    models, names = cfg["models"], cfgmod.area_names(cfg)
    K = len(names)
    M = len(models)
    mx = cfg["matrix"]
    trained = np.zeros((M, K), dtype=bool)
    for i, m in enumerate(models):
        trained[i, [cfgmod.area_index(names, a, f"model {m['id']}, trained_areas") for a in m["trained_areas"]]] = True
    ids = [m["id"] for m in models]

    def pr(name, layer):
        p = mx[name]
        out = np.zeros((M, K))
        for i, m in enumerate(models):
            for k in range(K):
                base = p["base"] - p["difficulty_scale"] * difficulty[k]
                v = base + p["size_bonus"][m["size"]] + (p["spec"] if trained[i, k] else -p["penalty"]) + p["noise"] * float(rng.normal(seed, rng.L_MATRIX, layer, i, k))
                v += mx["pr_cell_noise"] * float(rng.normal(seed, rng.L_MATRIX, layer + 10, i, k))
                out[i, k] = v
        return np.clip(out, 0.01, 0.995)

    P, R = pr("precision", 1), pr("recall", 2)
    free = np.array([m["id"] not in (mx.get("fixed") or {}) for m in models])
    F1 = f1_of(P, R)
    if free.any():                                          # rescale the free models to the F1 range of the simple interface
        lo, hi = F1[free].min(), F1[free].max()
        target = cfg["simple"]["perf_min"] + (F1 - lo) / max(hi - lo, 1e-9) * (cfg["simple"]["perf_max"] - cfg["simple"]["perf_min"])
        f = target / F1
        P[free], R[free] = np.clip(P[free] * f[free], 0.01, 0.995), np.clip(R[free] * f[free], 0.01, 0.995)
    for mid, fx in (mx.get("fixed") or {}).items():
        i = ids.index(mid)
        if len(fx["P"]) != K or len(fx["R"]) != K:
            raise ValueError(f"matrix.fixed.{mid}: P and R need one value per area ({K})")
        P[i], R[i] = np.array(fx["P"], float), np.array(fx["R"], float)
    F1 = f1_of(P, R)
    warnings = []
    if free.any() and (abs(F1[free].min() - cfg["simple"]["perf_min"]) > 0.04 or abs(F1[free].max() - cfg["simple"]["perf_max"]) > 0.04):
        warnings.append(f"F1 range [{F1[free].min():.2f}, {F1[free].max():.2f}] differs from [perf_min, perf_max] by more than 0.04 (clipping of P / R)")
    a_c, a_f = anchors(cfg)
    c, f = cfg["consistency"], cfg["confidence"]
    sg = cfg["signals"]
    nC = np.array([[float(rng.normal(seed, rng.L_MATRIX, 3, i, k)) for k in range(K)] for i in range(M)]) * (c["cell_noise_sd"] if sg["cell_noise"] else 0.0)      # signals.cell_noise off: the centres follow P and R exactly
    nF = np.array([[float(rng.normal(seed, rng.L_MATRIX, 4, i, k)) for k in range(K)] for i in range(M)]) * (f["cell_noise_sd"] if sg["cell_noise"] else 0.0)
    C = a_c + c["bp"] * P + c["br"] * R + nC
    Conf = a_f + f["bp"] * P + f["br"] * R + nF
    blind = np.zeros((M, K), dtype=bool)
    for i in range(M):
        for k in range(K):
            if sg["blind_cells"] and not trained[i, k] and float(rng.uniform(seed, rng.L_MATRIX, 5, i, k)) < cfg["blind"]["beta"]:
                blind[i, k] = True
    C = C + blind * cfg["blind"]["shift_sd"] * c["cell_noise_sd"]                  # a blind cell: consistency overestimates its quality; Conf is unaffected
    C, Conf = np.clip(C, 0.02, 0.98), np.clip(Conf, 0.02, 0.98)
    return {"P": P, "R": R, "F1": F1, "C": C, "Conf": Conf, "blind": blind, "trained": trained, "a_c": a_c, "a_f": a_f, "warnings": warnings}
