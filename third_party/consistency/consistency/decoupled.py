"""Frame-pair matching and the decoupled consistency signal.

For a transition ``t -> t+1`` (after class-wise NMS) this module computes:

* ``G``       - mean IoU over matched pairs (localisation agreement).
* ``delta_N`` - count fluctuation ``|N_t - N_t1| / max(N_t, N_t1, 1)``.
* ``m_star``  - matched fraction of the smaller frame, ``K / min(N_t, N_t1)``.
* ``m``       - ``m_star * (1 - delta_N)``, algebraically ``K / max(N_t, N_t1)``.
* ``R``       - ``G * m``, the uncompensated consistency.

``N_t`` and ``N_t1`` are box counts *after* NMS. ``status`` separates four regimes;
only ``MATCHED`` carries real quality scores, the others carry NaN for ``G``, ``m``
and ``R`` (no value is invented for "nothing to judge").

Matching is one-to-one (Hungarian assignment, ``scipy.optimize.linear_sum_assignment``)
on cost ``1 - IoU``. Pairs of different classes are blocked with a large sentinel cost
and a validity mask, so the solver can never pair them.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Sequence[float]  # [class_id, cx, cy, w, h, (conf)]

_BLOCKED = 1.0e9  # finite sentinel cost for gated-out cells (scipy rejects inf)


class Status(str, Enum):
    """Transition regime. Only MATCHED carries real consistency scores."""

    MATCHED = "MATCHED"
    NO_MATCH = "NO_MATCH"  # both frames non-empty, but no pair survives gating
    EMPTY_PAIR = "EMPTY_PAIR"  # both frames empty
    ONE_SIDED = "ONE_SIDED"  # exactly one frame empty


@dataclass(frozen=True)
class MatchPair:
    """A matched detection pair (indices into the *refined* box lists)."""

    box_t_idx: int
    box_t1_idx: int
    iou: float


@dataclass
class FramePairSignal:
    """Decoupled consistency signal for a single ``t -> t+1`` transition.

    Quality scores (``G``, ``m``, ``R``) are ``float('nan')`` unless
    ``status is Status.MATCHED``. ``m_star`` and ``delta_N`` are ``nan`` only
    when a side is empty (their defining ratios are undefined there).
    """

    G: float
    m_star: float
    delta_N: float
    m: float
    R: float
    K: int
    N_t: int
    N_t1: int
    status: Status
    matches: List[MatchPair] = field(default_factory=list)


@dataclass(frozen=True)
class MatchConfig:
    """Gating / matching configuration.

    Coordinates are YOLO-normalised (0-1). The spatial gate
    ``max(min_distance_px, distance_ratio * diag)`` uses ``min_distance_px = 100.0``,
    which in normalised space is never binding, so in practice a pair is admitted
    by class equality and the IoU floor ``min_iou`` alone. If pixel coordinates are
    supplied instead, the spatial gate becomes active.
    """

    apply_nms: bool = True
    nms_iou_threshold: float = 0.5
    nms_conf_threshold: float = 0.0
    min_iou: float = 0.05
    min_conf: float = 0.0
    distance_ratio: float = 1.5
    min_distance_px: float = 100.0
    matcher: str = "hungarian"  # "hungarian" | "greedy"


# --------------------------------------------------------------------------- #
# Geometry (vectorized)
# --------------------------------------------------------------------------- #
def iou_xywh(box_a: Box, box_b: Box) -> float:
    """Scalar IoU of two center-format boxes ``[cls, cx, cy, w, h, ...]``.

    Only the first five entries are used (the class id is ignored). Implemented in
    pure Python (no numpy allocation) because NMS calls it O(N^2) times per frame.
    """
    acx, acy, aw, ah = box_a[1], box_a[2], box_a[3], box_a[4]
    bcx, bcy, bw, bh = box_b[1], box_b[2], box_b[3], box_b[4]
    ax1, ay1, ax2, ay2 = acx - aw / 2.0, acy - ah / 2.0, acx + aw / 2.0, acy + ah / 2.0
    bx1, by1, bx2, by2 = bcx - bw / 2.0, bcy - bh / 2.0, bcx + bw / 2.0, bcy + bh / 2.0
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _iou_matrix(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    """Vectorized ``(N, M)`` IoU matrix for two arrays of ``[cx, cy, w, h]``."""
    if a_xywh.size == 0 or b_xywh.size == 0:
        return np.zeros((a_xywh.shape[0], b_xywh.shape[0]), dtype=float)
    ax1 = a_xywh[:, 0] - a_xywh[:, 2] / 2.0
    ay1 = a_xywh[:, 1] - a_xywh[:, 3] / 2.0
    ax2 = a_xywh[:, 0] + a_xywh[:, 2] / 2.0
    ay2 = a_xywh[:, 1] + a_xywh[:, 3] / 2.0
    bx1 = b_xywh[:, 0] - b_xywh[:, 2] / 2.0
    by1 = b_xywh[:, 1] - b_xywh[:, 3] / 2.0
    bx2 = b_xywh[:, 0] + b_xywh[:, 2] / 2.0
    by2 = b_xywh[:, 1] + b_xywh[:, 3] / 2.0

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = (a_xywh[:, 2] * a_xywh[:, 3])[:, None]
    area_b = (b_xywh[:, 2] * b_xywh[:, 3])[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


def _get_conf(box: Box) -> float:
    """Confidence of a box, defaulting to 1.0 when absent."""
    return float(box[5]) if len(box) > 5 else 1.0


# --------------------------------------------------------------------------- #
# Class-wise NMS
# --------------------------------------------------------------------------- #
def apply_nms_to_preds(
    preds: Sequence[Box],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0,
) -> Tuple[List[Box], List[int]]:
    """Class-wise greedy NMS. Returns ``(kept_boxes, kept_original_indices)``."""
    if not preds:
        return [], []
    # confidence filter, preserving original indices
    conf_kept = [(i, p) for i, p in enumerate(preds) if _get_conf(p) >= conf_threshold]
    if not conf_kept:
        return [], []
    # group by class
    by_class: dict = {}
    for i, p in conf_kept:
        by_class.setdefault(int(p[0]), []).append((i, p))

    kept: List[Tuple[int, Box]] = []
    for _cls, items in by_class.items():
        items_sorted = sorted(items, key=lambda ip: _get_conf(ip[1]), reverse=True)
        while items_sorted:
            i_best, best = items_sorted.pop(0)
            kept.append((i_best, best))
            survivors: List[Tuple[int, Box]] = []
            for i_other, other in items_sorted:
                if iou_xywh(best, other) < iou_threshold:
                    survivors.append((i_other, other))
            items_sorted = survivors
    kept.sort(key=lambda ip: ip[0])  # stable original ordering
    return [p for _i, p in kept], [i for i, _p in kept]


# --------------------------------------------------------------------------- #
# Refinement + matching
# --------------------------------------------------------------------------- #
def _refine(preds: Sequence[Box], cfg: MatchConfig) -> List[Box]:
    """Confidence filter + optional class-wise NMS, returning refined boxes."""
    if cfg.apply_nms:
        refined, _idx = apply_nms_to_preds(
            preds, iou_threshold=cfg.nms_iou_threshold,
            conf_threshold=cfg.nms_conf_threshold,
        )
        return refined
    return [p for p in preds if _get_conf(p) >= cfg.min_conf]


def _build_cost_and_validity(
    boxes_t: Sequence[Box], boxes_t1: Sequence[Box], cfg: MatchConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cost, valid, iou)`` matrices after class / spatial / IoU gating.

    ``cost[i, j] = 1 - iou`` where the pair is valid, else ``_BLOCKED``.
    ``valid[i, j]`` is the boolean gate result; ``iou`` is the raw IoU matrix.
    """
    n, m = len(boxes_t), len(boxes_t1)
    if n == 0 or m == 0:
        return (np.empty((n, m)), np.zeros((n, m), dtype=bool), np.zeros((n, m)))

    a = np.asarray([b[1:5] for b in boxes_t], dtype=float)
    b = np.asarray([b[1:5] for b in boxes_t1], dtype=float)
    iou = _iou_matrix(a, b)

    # class gate
    cls_t = np.asarray([int(x[0]) for x in boxes_t])[:, None]
    cls_t1 = np.asarray([int(x[0]) for x in boxes_t1])[None, :]
    same_class = cls_t == cls_t1

    # spatial gate: center distance <= max(min_distance_px, ratio * diag_source)
    cx_t, cy_t = a[:, 0][:, None], a[:, 1][:, None]
    cx_t1, cy_t1 = b[:, 0][None, :], b[:, 1][None, :]
    dist = np.hypot(cx_t - cx_t1, cy_t - cy_t1)
    diag_t = np.hypot(a[:, 2], a[:, 3])[:, None]
    dist_thresh = np.maximum(cfg.min_distance_px, cfg.distance_ratio * diag_t)
    spatial_ok = dist <= dist_thresh

    # IoU floor
    iou_ok = iou >= cfg.min_iou

    valid = same_class & spatial_ok & iou_ok
    cost = np.where(valid, 1.0 - iou, _BLOCKED)
    return cost, valid, iou


def _match_hungarian(
    cost: np.ndarray, valid: np.ndarray, iou: np.ndarray
) -> List[MatchPair]:
    row, col = linear_sum_assignment(cost)
    out: List[MatchPair] = []
    for i, j in zip(row, col):
        if valid[i, j]:
            out.append(MatchPair(int(i), int(j), float(iou[i, j])))
    return out


def _match_greedy(valid: np.ndarray, iou: np.ndarray) -> List[MatchPair]:
    cands: List[Tuple[float, int, int]] = []
    n, m = iou.shape
    for i in range(n):
        for j in range(m):
            if valid[i, j]:
                cands.append((float(iou[i, j]), i, j))
    cands.sort(key=lambda t: t[0], reverse=True)
    used_i: set = set()
    used_j: set = set()
    out: List[MatchPair] = []
    for iou_v, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append(MatchPair(i, j, iou_v))
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
_NAN = float("nan")


def compute_frame_pair_signal(
    preds_t: Sequence[Box],
    preds_t1: Sequence[Box],
    cfg: Optional[MatchConfig] = None,
) -> FramePairSignal:
    """Compute the decoupled :class:`FramePairSignal` for one transition.

    See the module docstring for the definitions and the NaN/status contract.
    """
    cfg = cfg or MatchConfig()
    refined_t = _refine(preds_t, cfg)
    refined_t1 = _refine(preds_t1, cfg)
    n_t, n_t1 = len(refined_t), len(refined_t1)

    # ---- Empty / one-sided regimes: nothing (or too little) to judge ----
    if n_t == 0 and n_t1 == 0:
        return FramePairSignal(
            G=_NAN, m_star=_NAN, delta_N=_NAN, m=_NAN, R=_NAN,
            K=0, N_t=0, N_t1=0, status=Status.EMPTY_PAIR, matches=[],
        )
    if n_t == 0 or n_t1 == 0:
        # exactly one side empty: count fluctuation is structurally 1.0, but
        # there is no correspondence to score -> quality scores NaN.
        return FramePairSignal(
            G=_NAN, m_star=_NAN, delta_N=1.0, m=_NAN, R=_NAN,
            K=0, N_t=n_t, N_t1=n_t1, status=Status.ONE_SIDED, matches=[],
        )

    # ---- Both sides non-empty: gate + match ----
    cost, valid, iou = _build_cost_and_validity(refined_t, refined_t1, cfg)
    if cfg.matcher == "greedy":
        matches = _match_greedy(valid, iou)
    else:
        matches = _match_hungarian(cost, valid, iou)

    K = len(matches)
    delta_N = abs(n_t - n_t1) / max(n_t, n_t1, 1)
    m_star = K / max(min(n_t, n_t1), 1)  # min >= 1 here since both non-empty
    m = m_star * (1.0 - delta_N)  # == K / max(n_t, n_t1)

    if K == 0:
        # both frames have detections but none correspond: a genuine NO_MATCH.
        # G and R are undefined (no matched pair to average), so they are NaN;
        # m_star/m are their true value 0.0.
        return FramePairSignal(
            G=_NAN, m_star=m_star, delta_N=delta_N, m=m, R=_NAN,
            K=0, N_t=n_t, N_t1=n_t1, status=Status.NO_MATCH, matches=[],
        )

    G = float(np.mean([mp.iou for mp in matches]))
    R = G * m
    return FramePairSignal(
        G=G, m_star=m_star, delta_N=delta_N, m=m, R=R,
        K=K, N_t=n_t, N_t1=n_t1, status=Status.MATCHED, matches=matches,
    )
