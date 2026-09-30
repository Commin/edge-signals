"""The locked monitor ``Gmc-full-1ch-incl`` for one pair of consecutive frames.

Name of the configuration:

* ``Gmc``  - localisation ``G`` is computed after global-shift compensation.
* ``full`` - every unmatched box counts in full (denominator ``max(N_t, N_t1, 1)``).
* ``1ch``  - a single composite alarm (no separate appearance / disappearance channels).
* ``incl`` - transitions with an empty side (``ONE_SIDED``) or no correspondence
  (``NO_MATCH``) are judged (alarm 1.0) rather than skipped; ``EMPTY_PAIR`` stays undefined.

Only in-memory box lists are consumed; no file access, image, or label is involved.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from .decoupled import (
    MatchConfig,
    MatchPair,
    Status,
    _refine,
    compute_frame_pair_signal,
    iou_xywh,
)
from .ego_motion import (
    compensate,
    estimate_global_shift,
    motion_magnitude,
)

Box = Sequence[float]


@dataclass(frozen=True)
class MonitorResult:
    """Output of one complete adjacent-frame monitoring operation."""

    status: str
    N_t: int
    N_t1: int
    K: int
    G: float
    G_mc: float
    dx: float
    dy: float
    motion_magnitude: float
    consistency: float
    alarm: float

    def to_dict(self) -> dict:
        return asdict(self)


def g_mc_frozen(
    refined_t: Sequence[Box],
    refined_t1: Sequence[Box],
    matches: Sequence[MatchPair],
) -> tuple:
    """Return ``(G_mc, dx, dy)`` while retaining the original assignment.

    Estimates the shared translation from the original matches, compensates frame
    ``t``, and re-scores the same pairs without a second assignment.
    """
    if not matches:
        return float("nan"), 0.0, 0.0
    dx, dy = estimate_global_shift(refined_t, refined_t1, matches)
    compensated_t = compensate(refined_t, dx, dy)
    ious = [
        iou_xywh(compensated_t[m.box_t_idx], refined_t1[m.box_t1_idx])
        for m in matches
    ]
    return float(np.mean(ious)), float(dx), float(dy)


def compute_locked_monitor(
    preds_t: Sequence[Box],
    preds_t1: Sequence[Box],
    cfg: Optional[MatchConfig] = None,
) -> MonitorResult:
    """Compute the complete ``Gmc-full-1ch-incl`` alarm for one transition.

    Computation order:

    1. class-wise NMS, gating, and Hungarian assignment;
    2. re-create the refined lists used by the match indices;
    3. estimate the median translation and re-score the original matches;
    4. count every unmatched detection in full;
    5. judge ONE_SIDED and NO_MATCH transitions as alarm 1; EMPTY_PAIR is undefined.
    """
    cfg = cfg or MatchConfig()
    sig = compute_frame_pair_signal(preds_t, preds_t1, cfg)

    refined_t = _refine(preds_t, cfg)
    refined_t1 = _refine(preds_t1, cfg)
    gmc, dx, dy = g_mc_frozen(refined_t, refined_t1, sig.matches)

    if sig.status is Status.EMPTY_PAIR:
        consistency = float("nan")
        alarm = float("nan")
    elif sig.status is Status.MATCHED and math.isfinite(gmc):
        # full mismatch counting gives N_t and N_t1 unchanged in the denominator.
        consistency = gmc * sig.K / max(sig.N_t, sig.N_t1, 1)
        alarm = 1.0 - consistency
    else:
        # Gmc-full-1ch-incl: NO_MATCH and ONE_SIDED are judged as zero consistency.
        consistency = 0.0
        alarm = 1.0

    return MonitorResult(
        status=sig.status.value,
        N_t=sig.N_t,
        N_t1=sig.N_t1,
        K=sig.K,
        G=float(sig.G),
        G_mc=float(gmc),
        dx=float(dx),
        dy=float(dy),
        motion_magnitude=float(motion_magnitude(dx, dy)),
        consistency=float(consistency),
        alarm=float(alarm),
    )
