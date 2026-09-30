"""platform_motion (context, pair): the global image shift |dx|, |dy| estimated by the consistency package. Measure only."""
from __future__ import annotations

import math

from ..base import PairRecord, Signal
from ..registry import register
from .consistency import _monitor


@register
class PlatformMotion(Signal):
    name = "platform_motion"
    version = "1"
    family = "context"
    granularity = "pair"
    requires = frozenset({"preds"})
    higher_is_healthier = False        # larger motion = more disturbance (context, not a health measure)
    label_free = True

    def __init__(self, params=None, reference=None):
        super().__init__(params, reference)
        self._fn = _monitor()

    def update(self, record: PairRecord):
        res = record.cache.get("consistency_result")
        if res is None:                # shared with the consistency signal when both are enabled
            res = self._fn(record.prev.boxes.tolist(), record.cur.boxes.tolist())
            record.cache["consistency_result"] = res
        d = res.to_dict()
        extras = {"status": d["status"], "dx": d["dx"], "dy": d["dy"], "abs_dx": abs(d["dx"]), "abs_dy": abs(d["dy"])}
        if d["status"] != "MATCHED":   # the shift is only estimated from matched boxes (otherwise 0.0 by construction)
            return self.invalid(record, "no_matched_pairs", extras)
        return self.value(record, math.hypot(d["dx"], d["dy"]), extras)
