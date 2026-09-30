"""Offline detection metrics (LABEL-REQUIRED): single-class AP (ultralytics matching / ap_per_class) and recall by box size."""
from __future__ import annotations

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_tp(iou):
    """TP matrix (n_pred, 10 IoU thresholds), one class."""
    correct = np.zeros((iou.shape[1], len(IOUV)), bool)
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def ap_single_class(frames):
    """frames: [(gt_xyxy, pred_xyxy, pred_conf)] -> (mAP50, mAP50-95), or (None, None) without ground truth."""
    from ultralytics.utils.metrics import ap_per_class
    n_gt = sum(len(g) for g, _, _ in frames)
    if n_gt == 0:
        return None, None
    tp, conf = [], []
    for g, pb, pf in frames:
        if len(pb):
            tp.append(match_tp(box_iou(g, pb)) if len(g) else np.zeros((len(pb), 10), bool))
            conf.append(pf)
    if not tp:
        return 0.0, 0.0
    r = ap_per_class(np.concatenate(tp), np.concatenate(conf), np.zeros(sum(len(c) for c in conf)), np.zeros(n_gt), plot=False)
    ap = r[5]
    return float(ap[:, 0].mean()), float(ap.mean())


def recall_by_size(frames, bins, iou_thr, conf_min):
    """One-to-one greedy recall; predictions with conf >= conf_min; sizes = sqrt(box area) in px, bins [lo, hi]."""
    lo, hi = bins
    keys = [f"<{lo}", f"{lo}-{hi}", f">{hi}"]
    tot = {k: [0, 0] for k in keys}
    for g, pb, pf in frames:
        k_ = pf >= conf_min
        pb2 = pb[k_][np.argsort(-pf[k_])] if len(pb) else pb
        iou = box_iou(g, pb2)
        hit, used = np.zeros(len(g), bool), np.zeros(len(pb2), bool)
        for v, i, j in sorted(((iou[i, j], i, j) for i in range(len(g)) for j in range(len(pb2)) if iou[i, j] >= iou_thr), reverse=True):
            if not hit[i] and not used[j]:
                hit[i], used[j] = True, True
        s = np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) if len(g) else np.zeros(0)
        for k, sel in zip(keys, (s < lo, (s >= lo) & (s <= hi), s > hi)):
            tot[k][0] += int(sel.sum())
            tot[k][1] += int(hit[sel].sum())
    out = {k: {"n_gt": n, "recall": (m / n if n else None)} for k, (n, m) in tot.items()}
    n = sum(v[0] for v in tot.values())
    out["all"] = {"n_gt": n, "recall": (sum(v[1] for v in tot.values()) / n if n else None)}
    return out
