"""confidence (quality, frame): mean confidence of the kept boxes of a frame. Measure only by default."""
from __future__ import annotations

from ..base import FrameRecord, Signal
from ..registry import register


@register
class Confidence(Signal):
    name = "confidence"
    version = "1"
    family = "quality"
    granularity = "frame"
    requires = frozenset({"preds"})
    higher_is_healthier = True
    label_free = True

    def update(self, record: FrameRecord):
        b = record.boxes
        if len(b) == 0:
            return self.invalid(record, "no_boxes", {"n_boxes": 0})
        return self.value(record, float(b[:, 5].mean()), {"n_boxes": int(len(b))})
