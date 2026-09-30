"""Global camera-motion estimation and compensation from box metadata.

Camera motion drags every box together and depresses the raw matched IoU ``G`` even
when the scene itself is unchanged. The motion is estimated directly from the
detector's own matched boxes, so no image is needed:

* :func:`estimate_global_shift` - the **median** matched-centre displacement, taken
  independently per axis. The median has a breakdown point of 1/2: up to half of the
  matched pairs may move independently of the camera before the estimate is corrupted.
* :func:`compensate` - shift boxes by the estimated motion.
* :func:`motion_magnitude` - ``hypot(dx, dy)``.
"""

from typing import List, Sequence, Tuple

import numpy as np

from .decoupled import MatchPair

Box = Sequence[float]


def estimate_global_shift(
    boxes_t: Sequence[Box], boxes_t1: Sequence[Box], matches: Sequence[MatchPair]
) -> Tuple[float, float]:
    """Median matched-center displacement ``(dx, dy)`` from t to t+1.

    Returns ``(0.0, 0.0)`` when there are no matches.
    """
    if not matches:
        return 0.0, 0.0
    dxs = [boxes_t1[m.box_t1_idx][1] - boxes_t[m.box_t_idx][1] for m in matches]
    dys = [boxes_t1[m.box_t1_idx][2] - boxes_t[m.box_t_idx][2] for m in matches]
    return float(np.median(dxs)), float(np.median(dys))


def motion_magnitude(dx: float, dy: float) -> float:
    return float(np.hypot(dx, dy))


def compensate(boxes: Sequence[Box], dx: float, dy: float, ds: float = 0.0) -> List[List[float]]:
    """Translate boxes by ``(dx, dy)`` and, optionally, scale each side by ``exp(ds/2)``.

    ``ds`` is a global log-area change; the locked monitor uses ``ds = 0`` (translation only).
    """
    scale = float(np.exp(ds / 2.0))
    out: List[List[float]] = []
    for b in boxes:
        nb = list(b)
        nb[1] = b[1] + dx
        nb[2] = b[2] + dy
        nb[3] = b[3] * scale
        nb[4] = b[4] * scale
        out.append(nb)
    return out
