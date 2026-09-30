"""Window statistic: aggregator, comparator, change - per signal. Never fills invalid values."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

AGGREGATORS = ("mean", "median", "worst10")
COMPARATORS = ("previous", "anchor", "rolling", "reference")
CHANGES = ("difference", "ratio")


def aggregate(values, name: str, higher_is_healthier: bool) -> float:
    """mean | median | worst10 (mean of the worst 10 % of values: the lowest if higher is healthier, else the highest)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        raise ValueError("no values to aggregate")
    if name == "mean":
        return float(v.mean())
    if name == "median":
        return float(np.median(v))
    if name == "worst10":
        k = max(1, math.ceil(0.10 * v.size))
        s = np.sort(v)
        return float((s[:k] if higher_is_healthier else s[-k:]).mean())
    raise ValueError(f"unknown aggregator {name}")


@dataclass
class WindowStat:
    signal: str
    window_id: int
    clip_id: Optional[str]
    t_start: float
    t_end: float
    n_items: int
    n_valid: int
    statistic: Optional[float]
    valid: bool
    invalid_reason: Optional[str]
    reference_kind: str                  # previous | anchor | rolling | reference
    reference: Optional[float]
    change: Optional[float]              # degradation: > 0 means worse than the reference period
    change_kind: str                     # difference | ratio
    aggregator: str
    window_transitions: int
    higher_is_healthier: bool
    model_version: str = ""
    device_id: str = ""
    reference_id: Optional[str] = None
    window_seconds: Optional[float] = None       # window_transitions / frame rate, when the frame rate is known
    comparisons: Optional[dict] = None           # every registered comparator: {key: {reference, change, reason}}

    def to_dict(self):
        return dict(self.__dict__)


class Comparison:
    """The reference period of one comparator and the change of a window statistic against it (positive = worse)."""

    def __init__(self, signal, kind: str, aggregator: str, W: int, change: str, M: int):
        self.signal, self.kind, self.aggregator, self.W, self.change, self.M = signal, kind, aggregator, W, change, M
        self.h = signal.higher_is_healthier
        self.prev_stat: Optional[float] = None
        self.anchor: Optional[float] = None
        self.hist: List[float] = []
        self._level: Optional[float] = None

    @property
    def key(self) -> str:
        return f"rolling:{self.M}" if self.kind == "rolling" else self.kind

    def reset(self):
        """After a trigger: the anchor and the rolling history restart; 'previous' restarts too."""
        self.anchor, self.hist, self.prev_stat = None, [], None

    def reference(self):
        c = self.kind
        if c == "previous":
            return self.prev_stat, None if self.prev_stat is not None else "no_previous_window"
        if c == "anchor":
            return self.anchor, None if self.anchor is not None else "anchor_window"
        if c == "rolling":
            if len(self.hist) < self.M:
                return None, "rolling_history_incomplete"
            return float(np.mean(self.hist[-self.M:])), None
        if c == "reference":
            if self._level is None:
                self._level = self.signal.reference_level(self.aggregator, self.W)
            return self._level, None if self._level is not None else "no_reference_level"
        raise ValueError(c)

    def compare(self, stat: Optional[float]):
        """-> (reference, change, reason). change is None when it is not defined (reason says why)."""
        ref, rreason = self.reference()
        if stat is None:
            return ref, None, "no_statistic"
        if ref is None:
            return None, None, rreason
        if self.change == "difference":
            return ref, ((ref - stat) if self.h else (stat - ref)), None
        if ref == 0:
            return ref, None, "zero_reference"
        return ref, ((1 - stat / ref) if self.h else (stat / ref - 1)), None

    def update(self, stat: Optional[float]):
        self.prev_stat = stat
        if stat is not None:
            self.hist.append(stat)
            if self.anchor is None:
                self.anchor = stat


class WindowTracker:
    """Collects the values of one signal into non-overlapping windows of `window_transitions` items and compares each
    window statistic with its reference period(s). Windows never cross a clip change; the comparison state does.
    The signal's own comparator is the primary one (logged in reference / change); trigger rules may register more."""

    def __init__(self, signal, cfg: dict, fps: Optional[float] = None):
        self.signal = signal
        self.W = int(cfg["window_transitions"])
        self.window_seconds = self.W / fps if fps else None
        self.aggregator = cfg["aggregator"]
        self.comparator = cfg["comparator"]
        self.M = int(cfg.get("rolling_windows", 3))
        self.change = cfg["change"]
        self.min_valid = float(cfg.get("min_valid_fraction", 0.5))
        self.h = signal.higher_is_healthier
        self.buf: list = []
        self.window_id = 0
        self.dropped_partial = 0
        self.comparisons = {}
        self.primary = self.add_comparison(self.comparator, self.M, self.change)

    def add_comparison(self, kind: str, M: Optional[int] = None, change: Optional[str] = None) -> str:
        c = Comparison(self.signal, kind, self.aggregator, self.W, change or self.change, int(M or self.M))
        key = c.key + ("" if (change or self.change) == self.change else f"/{change}")
        self.comparisons.setdefault(key, c)
        return key

    # ---------------------------------------------------------------- state
    def reset_reference(self):
        for c in self.comparisons.values():
            c.reset()

    def close_clip(self):
        """A clip change: the incomplete window is dropped (windows never cross a clip); the comparison state stays."""
        if self.buf:
            self.dropped_partial += 1
        self.buf = []

    # ---------------------------------------------------------------- input
    def push(self, sv, clip_id=None) -> Optional[WindowStat]:
        self.buf.append((sv, clip_id))
        if len(self.buf) < self.W:
            return None
        items, self.buf = self.buf, []
        return self._close(items)

    def _close(self, items) -> WindowStat:
        vals = [sv for sv, _ in items]
        ok = [sv.value for sv in vals if sv.valid]
        stat, reason = None, None
        if len(ok) / len(vals) < self.min_valid or not ok:
            reason = "too_few_valid_values"
        else:
            stat = aggregate(ok, self.aggregator, self.h)
        comps = {}
        for key, c in self.comparisons.items():
            ref, chg, why = c.compare(stat)
            comps[key] = {"reference": ref, "change": chg, "reason": why}
        p = comps[self.primary]
        ws = WindowStat(signal=self.signal.name, window_id=self.window_id, clip_id=items[-1][1], t_start=vals[0].t,
                        t_end=vals[-1].t, n_items=len(vals), n_valid=len(ok), statistic=stat, valid=stat is not None,
                        invalid_reason=reason, reference_kind=self.comparator, reference=p["reference"], change=p["change"],
                        change_kind=self.change, aggregator=self.aggregator, window_transitions=self.W,
                        higher_is_healthier=self.h, model_version=vals[-1].model_version, device_id=vals[-1].device_id,
                        reference_id=vals[-1].reference_id, window_seconds=self.window_seconds, comparisons=comps)
        if p["change"] is None and stat is not None:
            ws.invalid_reason = f"no_change:{p['reason']}"        # the statistic is valid, the comparison is not defined
        self.window_id += 1
        for c in self.comparisons.values():                          # comparison state (crosses clip changes)
            c.update(stat)
        return ws
