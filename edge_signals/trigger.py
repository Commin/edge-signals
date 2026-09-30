"""Trigger rules and adaptation requests.

Every fired rule emits an **adaptation request** (an event that the decision layer acts on; it is not a retraining decision).
Each rule keeps its OWN state (comparison reference, anchor, counters, flag history, cooldown); a firing rule resets only its own state.

Rule types
* relative_drop: fire when the window's degradation against the rule's comparator exceeds delta for n_consecutive windows in a row.
    step      = comparator previous, N = 1                              (stage: screening)
    sustained = comparator anchor (the window after the rule's last fire, or the stream start), N >= 2   (stage: confirmation)
* rate: fire when at least min_flagged of the last `windows` windows were FLAGGED. A window is flagged when the flag rule's condition holds
  (its comparator and delta) - the condition, not the fired request. The rate rule evaluates that condition on its own state.   (stage: confirmation)
The four prioritised trigger types (critical, cumulative, preventive, periodic) remain hooks and refuse to run if enabled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import comb

import numpy as np
from typing import Dict, List, Optional

from .windows import Comparison

STAGES = ("screening", "confirmation")


@dataclass
class Fired:
    """What a rule reports when it fires (the engine adds the evidence and the context)."""
    rule: str
    rule_type: str
    stage: str
    signal: str
    ws: object
    comparator: str
    reference: Optional[float]
    change: Optional[float]
    delta: float
    n_consecutive: int
    cooldown_windows: int
    flags: Optional[dict] = None


@dataclass
class AdaptationRequest:
    request_id: str
    stage: str                          # screening | confirmation
    rule: str
    rule_type: str
    signal: str
    window_id: int
    clip_id: Optional[str]
    t: float
    comparator: str
    statistic_before: Optional[float]   # the reference period's level (previous window / anchor)
    statistic_after: Optional[float]    # this window's statistic
    drop: Optional[float]               # degradation, > 0 = worse (difference or ratio form)
    drop_kind: str
    delta: float
    n_consecutive: int
    cooldown_windows: int
    p_value: Optional[float]            # empirical: share of healthy windows whose drop was at least this large
    p_value_note: Optional[str]
    p_value_floor: Optional[float]      # smallest possible p-value = 1 / (n_healthy + 1): p at this value means "at least as extreme as every healthy window"
    robust_z: Optional[float]           # drop / (1.4826 x MAD of the healthy change distribution): a scale-free size of the drop
    robust_z_note: Optional[str]
    flag_rate: Optional[dict]           # over the last `windows` windows: {windows, flagged, rate, healthy_flag_probability, binomial_p}
    context: dict = field(default_factory=dict)      # latest values of the measured signals - context only, never used by a rule
    model_version: str = ""
    device_id: str = ""
    video: Optional[str] = None         # video input: where in the stream the request was raised (the last frame of its window) ...
    frame_index: Optional[int] = None   # ... and the ORIGINAL frame index of that frame

    def to_dict(self):
        d = dict(self.__dict__)
        d["kind"] = "adaptation_request"
        return d


class Rule:
    type = ""
    stage = "screening"

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.stage = cfg.get("stage", "screening")


class _State:
    def __init__(self, cmp: Comparison, maxlen: int = 0):
        self.cmp, self.consec, self.cool = cmp, 0, 0
        self.flags = deque(maxlen=maxlen) if maxlen else None

    def reset(self):
        self.cmp.reset()
        self.consec = 0
        if self.flags is not None:
            self.flags.clear()


class RelativeDrop(Rule):
    type = "relative_drop"

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.delta = float(cfg["delta"])
        self.N = int(cfg["n_consecutive"])
        self.C = int(cfg["cooldown_windows"])
        self.comparator = cfg["comparator"]
        self.M = int(cfg.get("rolling_windows") or 3)
        self.state: Dict[str, _State] = {}

    def register(self, signal, tracker):
        self.state[signal] = _State(Comparison(tracker.signal, self.comparator, tracker.aggregator, tracker.W, tracker.change, self.M))

    def on_window(self, ws) -> Optional[Fired]:
        st = self.state[ws.signal]
        ref, change, _ = st.cmp.compare(ws.statistic)
        st.cmp.update(ws.statistic)
        if st.cool > 0:
            st.cool -= 1
            st.consec = 0
            return None
        ex = ws.valid and change is not None and change > self.delta
        st.consec = st.consec + 1 if ex else 0
        if st.consec < self.N:
            return None
        f = Fired(self.name, self.type, self.stage, ws.signal, ws, self.comparator, ref, change, self.delta, self.N, self.C)
        st.reset()
        st.cool = self.C
        return f


class Rate(Rule):
    type = "rate"

    def __init__(self, name, cfg, rules_by_name):
        super().__init__(name, cfg)
        src = rules_by_name.get(cfg["flag_rule"])
        if src is None or not isinstance(src, RelativeDrop):
            raise ValueError(f"rate rule {name}: flag_rule '{cfg['flag_rule']}' must be an enabled relative_drop rule")
        self.flag_comparator, self.flag_delta, self.flag_M = src.comparator, src.delta, src.M
        self.windows = int(cfg["windows"])
        self.k = int(cfg["min_flagged"])
        self.C = int(cfg.get("cooldown_windows", 0))
        self.state: Dict[str, _State] = {}

    def register(self, signal, tracker):
        self.state[signal] = _State(Comparison(tracker.signal, self.flag_comparator, tracker.aggregator, tracker.W, tracker.change, self.flag_M),
                                    self.windows)

    def on_window(self, ws) -> Optional[Fired]:
        st = self.state[ws.signal]
        ref, change, _ = st.cmp.compare(ws.statistic)
        st.cmp.update(ws.statistic)
        flag = bool(ws.valid and change is not None and change > self.flag_delta)
        st.flags.append(flag)
        if st.cool > 0:
            st.cool -= 1
            return None
        n = sum(st.flags)
        if n < self.k:
            return None
        f = Fired(self.name, self.type, self.stage, ws.signal, ws, self.flag_comparator, ref, change, self.flag_delta, 1, self.C,
                  flags={"windows": len(st.flags), "flagged": n})
        st.reset()
        st.cool = self.C
        return f


class _Hook(Rule):
    """A prioritised trigger type kept as a hook: present in the code, disabled in the config."""

    def __init__(self, name, cfg, *a):
        super().__init__(name, cfg)
        raise NotImplementedError(f"trigger rule '{self.type}' is a hook and is not implemented; keep it disabled")


class Critical(_Hook):
    type = "critical"


class Cumulative(_Hook):
    type = "cumulative"


class Preventive(_Hook):
    type = "preventive"


class Periodic(_Hook):
    type = "periodic"


RULES = {c.type: c for c in (RelativeDrop, Rate, Critical, Cumulative, Preventive, Periodic)}
PRIORITY = ("critical", "cumulative", "preventive", "periodic")      # the four prioritised types (hooks)


def binom_tail(n: int, k: int, q: float) -> float:
    """P(X >= k) for X ~ Binomial(n, q)."""
    return float(sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1)))


class TriggerEngine:
    """Owns the rules, the flag history used as evidence, and the healthy-change distribution (p-values)."""

    def __init__(self, trigger_cfg: dict, trigger_signals: List[str], healthy=None):
        self.signals = set(trigger_signals)
        self.healthy = healthy                                   # ReferenceBundle of healthy drops, or None
        self.rules: List[Rule] = []
        by_name: Dict[str, Rule] = {}
        rules_cfg = trigger_cfg.get("rules") or {}
        for name, rc in rules_cfg.items():                       # relative_drop rules first so rate rules can refer to them
            typ = rc.get("type", name)
            if typ not in RULES:
                raise ValueError(f"unknown trigger rule type {typ} (rule {name})")
            if rc.get("enabled", False) and typ != "rate":
                r = RULES[typ](name, rc)
                self.rules.append(r)
                by_name[name] = r
        for name, rc in rules_cfg.items():
            if rc.get("type", name) == "rate" and rc.get("enabled", False):
                self.rules.append(Rate(name, rc, by_name))
        ev = trigger_cfg.get("evidence") or {}
        self.ev_windows = int(ev.get("windows", 12))
        flag_rule = by_name.get(ev.get("flag_rule", "step")) or next((r for r in self.rules if isinstance(r, RelativeDrop)), None)
        self.flag_rule = flag_rule
        self.ev_state: Dict[str, _State] = {}

    def register(self, trackers: dict):
        for s in self.signals:
            if s not in trackers:
                continue
            for r in self.rules:
                r.register(s, trackers[s])
            if self.flag_rule is not None:                       # evidence only: never reset, never read by a rule
                t = trackers[s]
                self.ev_state[s] = _State(Comparison(t.signal, self.flag_rule.comparator, t.aggregator, t.W, t.change, self.flag_rule.M), self.ev_windows)

    # ------------------------------------------------------------------ per window
    def on_window(self, ws) -> List[Fired]:
        if ws.signal not in self.signals:
            return []
        s = ws.signal
        ev = self.ev_state.get(s)
        if ev is not None:
            _, chg, _ = ev.cmp.compare(ws.statistic)
            ev.cmp.update(ws.statistic)
            ev.flags.append(bool(ws.valid and chg is not None and chg > self.flag_rule.delta))
        return [f for f in (r.on_window(ws) for r in self.rules) if f is not None]

    # ------------------------------------------------------------------ evidence
    def p_value(self, comparator: str, drop: Optional[float]):
        if drop is None:
            return None, "no_drop"
        if self.healthy is None or comparator not in self.healthy.arrays:
            return None, "no_healthy_distribution"
        h = self.healthy.arrays[comparator]
        return float((1 + (h >= drop).sum()) / (len(h) + 1)), None

    def robust_z(self, comparator: str, drop: Optional[float]):
        """drop / (1.4826 x MAD) with the median absolute deviation of the same healthy distribution the p-value uses."""
        if drop is None:
            return None, "no_drop"
        if self.healthy is None or comparator not in self.healthy.arrays:
            return None, "no_healthy_distribution"
        h = self.healthy.arrays[comparator]
        mad = float(np.median(np.abs(h - np.median(h))))
        if mad <= 0:
            return None, "zero_mad"
        return float(drop / (1.4826 * mad)), None

    def p_floor(self, comparator: str) -> Optional[float]:
        if self.healthy is None or comparator not in self.healthy.arrays:
            return None
        return 1.0 / (len(self.healthy.arrays[comparator]) + 1)

    def flag_rate(self, signal: str):
        ev = self.ev_state.get(signal)
        if ev is None or not len(ev.flags):
            return None
        n, m = sum(ev.flags), len(ev.flags)
        q = None
        if self.healthy is not None and self.flag_rule.comparator in self.healthy.arrays:
            h = self.healthy.arrays[self.flag_rule.comparator]
            q = float((h > self.flag_rule.delta).mean())
        return {"windows": m, "flagged": n, "rate": n / m, "healthy_flag_probability": q,
                "binomial_p": binom_tail(m, n, q) if q is not None else None}

    def request(self, f: Fired, context: dict, seq: int, position=(None, None)) -> AdaptationRequest:
        ws = f.ws
        p, note = self.p_value(f.comparator, f.change)
        rz, rz_note = self.robust_z(f.comparator, f.change)
        fr = self.flag_rate(f.signal)
        return AdaptationRequest(request_id=f"{f.rule}-{f.signal}-{ws.window_id}", stage=f.stage, rule=f.rule, rule_type=f.rule_type,
                                 signal=f.signal, window_id=ws.window_id, clip_id=ws.clip_id, t=ws.t_end, comparator=f.comparator,
                                 statistic_before=f.reference, statistic_after=ws.statistic, drop=f.change, drop_kind=ws.change_kind,
                                 delta=f.delta, n_consecutive=f.n_consecutive, cooldown_windows=f.cooldown_windows, p_value=p,
                                 video=position[0], frame_index=position[1], p_value_note=note, p_value_floor=self.p_floor(f.comparator), robust_z=rz, robust_z_note=rz_note, flag_rate=fr, context=context, model_version=ws.model_version, device_id=ws.device_id)
