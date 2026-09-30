"""consistency (quality, pair): the released label-free motion-compensated consistency package, vendored unchanged
in third_party/consistency/ (locked cell Gmc-full-1ch-incl). This module only adapts records and never alters its maths."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from ..base import PairRecord, Signal
from ..registry import register

VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "consistency"


def vendored_commit() -> str:
    m = re.search(r"commit\s*:\s*([0-9a-f]{40})", (VENDOR / "VENDORED.txt").read_text())
    return m.group(1)


def _monitor():
    """Import compute_locked_monitor from the vendored copy (never from anywhere else)."""
    try:
        import consistency
    except ImportError:
        sys.path.append(str(VENDOR))
        import consistency
    if VENDOR.resolve() not in Path(consistency.__file__).resolve().parents:
        raise ImportError(f"'consistency' is imported from {consistency.__file__}, not from the vendored copy {VENDOR}")
    from consistency.monitor import compute_locked_monitor
    return compute_locked_monitor


@register
class Consistency(Signal):
    name = "consistency"
    family = "quality"
    granularity = "pair"
    requires = frozenset({"preds"})
    higher_is_healthier = True
    label_free = True

    def __init__(self, params=None, reference=None):
        super().__init__(params, reference)
        self._fn = _monitor()
        self.version = f"gmc-full-1ch-incl@{vendored_commit()[:7]}"

    def update(self, record: PairRecord):
        res = record.cache.get("consistency_result")
        if res is None:
            res = self._fn(record.prev.boxes.tolist(), record.cur.boxes.tolist())
            record.cache["consistency_result"] = res
        d = res.to_dict()
        extras = {k: d[k] for k in ("alarm", "status", "G", "G_mc", "dx", "dy", "K", "N_t", "N_t1")}
        if d["status"] == "EMPTY_PAIR":
            return self.invalid(record, "EMPTY_PAIR", extras)      # the package leaves it undefined; never filled
        return self.value(record, d["consistency"], extras)
