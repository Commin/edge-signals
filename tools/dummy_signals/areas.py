"""Area timelines: the global semi-Markov timeline, the delayed view of each edge, local episodes, ramps between areas, and the scripted routes."""
from __future__ import annotations

import math

import numpy as np

from . import rng


def global_timeline(seed, K, tl, t_start, t_end):
    """Segments [(t0, t1, area)] covering [t_start, t_end]. Segment i is a function of (seed, i) and of the previous area only."""
    segs, t, prev, i = [], t_start, None, 0
    if tl.get("fixed_segments"):                       # explicit sequence, cycled; its time origin is t = 0
        fs = tl["fixed_segments"]
        period = sum(float(d) for _, d in fs)
        t = -period * math.ceil(max(-t_start, 0.0) / period)           # whole periods before t = 0: the sequence starts at t = 0
        while t < t_end:
            for (r, d) in fs:
                segs.append((t, t + float(d), int(r)))
                t += float(d)
        return segs
    while t < t_end:
        if prev is None or K == 1:
            r = int(rng.uniform(seed, rng.L_TIMELINE, i, 0) * K) % K
        else:
            r = (prev + 1 + int(rng.uniform(seed, rng.L_TIMELINE, i, 0) * (K - 1))) % K     # uniform among the other areas
        d = float(np.clip(tl["median_s"] * math.exp(tl["sigma"] * float(rng.normal(seed, rng.L_TIMELINE, i, 1))), tl["min_s"], tl["max_s"]))
        segs.append((t, t + d, r))
        t, prev, i = t + d, r, i + 1
    return segs


def local_episodes(seed, edge_idx, K, ep, t_end):
    """Poisson process of episodes [(t0, t1, area)] seen by one edge only."""
    out, t, i = [], 0.0, 0
    rate = ep["rate_per_hour"] / 3600.0
    if rate <= 0:
        return out
    while True:
        t += -math.log(float(rng.uniform(seed, rng.L_EPISODE, edge_idx, i, 0))) / rate
        if t >= t_end:
            return out
        r = int(rng.uniform(seed, rng.L_EPISODE, edge_idx, i, 1) * K) % K
        d = ep["median_s"] * math.exp(ep["sigma"] * float(rng.normal(seed, rng.L_EPISODE, edge_idx, i, 2)))
        out.append((t, t + max(d, 10.0), r))
        i += 1


def overlay(base, episodes):
    """base: [(t0, t1, area, origin)]; each episode (in start order) overrides its interval."""
    segs = list(base)
    for (a, b, r) in episodes:
        new = []
        for (s0, s1, sr, so) in segs:
            if s1 <= a or s0 >= b:
                new.append((s0, s1, sr, so))
                continue
            if s0 < a:
                new.append((s0, a, sr, so))
            if s1 > b:
                new.append((b, s1, sr, so))
        new.append((a, b, r, "local"))
        segs = sorted(new)
    merged = []
    for s in segs:
        if merged and merged[-1][2] == s[2] and merged[-1][3] == s[3] and abs(merged[-1][1] - s[0]) < 1e-9:
            merged[-1] = (merged[-1][0], s[1], s[2], s[3])
        else:
            merged.append(s)
    return merged


class EdgeTimeline:
    """area(e, t) as segments, plus the mix (from, to, s) that implements ramps between neighbouring areas."""

    def __init__(self, segs, ramp_s):
        self.segs = segs
        self.starts = np.array([s[0] for s in segs])
        self.labels = np.array([s[2] for s in segs], dtype=int)
        n = len(segs)
        self.ramp = np.zeros(n)                      # ramp[j]: length of the ramp centred on the boundary between segment j-1 and j (ramp[0] unused)
        for j in range(1, n):
            self.ramp[j] = min(ramp_s, segs[j][1] - segs[j][0], segs[j - 1][1] - segs[j - 1][0])

    def mix(self, t):
        """-> (from area, to area, s in [0, 1], label). s is a smoothstep over the ramp; label = the area that is dominant at t."""
        t = np.asarray(t, dtype=float)
        n = len(self.segs)
        j = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, n - 1)
        a = self.labels[j].copy()
        b = a.copy()
        s = np.zeros_like(t)
        nxt = np.minimum(j + 1, n - 1)
        rn = self.ramp[nxt] * (j + 1 < n)
        bn = self.starts[nxt]
        in_next = (j + 1 < n) & (rn > 0) & (t >= bn - rn / 2)
        x = np.where(in_next, (t - (bn - rn / 2)) / np.where(rn > 0, rn, 1), 0)
        b = np.where(in_next, self.labels[nxt], b)
        s = np.where(in_next, x, s)
        rc = self.ramp[j]
        bc = self.starts[j]
        in_prev = (j >= 1) & (rc > 0) & (t < bc + rc / 2) & ~in_next
        x2 = np.where(in_prev, (t - (bc - rc / 2)) / np.where(rc > 0, rc, 1), 0)
        a = np.where(in_prev, self.labels[np.maximum(j - 1, 0)], a)
        b = np.where(in_prev, self.labels[j], b)
        s = np.where(in_prev, x2, s)
        s = np.clip(s, 0.0, 1.0)
        s = s * s * (3 - 2 * s)
        label = np.where(s >= 0.5, b, a)
        return a, b, s, label

    def label_at(self, t):
        return self.mix(t)[3]

    def shift(self, t, mi):
        """No transition effect on a random timeline."""
        return np.zeros_like(np.asarray(t, dtype=float))


class RouteTimeline:
    """A scripted route: stay steps (one area) and move steps (a linear blend from the previous area to the next one). Same interface as EdgeTimeline.

    steps: [{"kind": "stay" | "move", "t0", "t1", "area" (index; the target of a move), "from_area" (index; moves only)}], contiguous from t = 0.
    episodes: [(t0, t1, area)] of random local episodes; on a route they are abrupt (the area is replaced for that time).
    """

    def __init__(self, steps, episodes=(), common=None, dev=None):
        self.steps = steps
        n = len(steps)
        self.t_common = np.zeros(n) if common is None else np.asarray(common, float)                 # transition effect: per step, the common part ...
        self.t_dev = np.zeros((n, 1)) if dev is None else np.asarray(dev, float)                     # ... and the deviation of each model (steps x models); zero = no effect
        self.starts = np.array([s["t0"] for s in steps])
        self.ends = np.array([s["t1"] for s in steps])
        self.to = np.array([s["area"] for s in steps], dtype=int)
        self.move = np.array([s["kind"] == "move" for s in steps])
        self.frm = np.array([s.get("from_area", s["area"]) for s in steps], dtype=int)
        self.episodes = list(episodes)

    def mix(self, t):
        t = np.asarray(t, dtype=float)
        j = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.steps) - 1)
        mv = self.move[j]
        a = np.where(mv, self.frm[j], self.to[j])
        b = self.to[j].copy()
        s = np.where(mv, np.clip((t - self.starts[j]) / np.maximum(self.ends[j] - self.starts[j], 1e-9), 0.0, 1.0), 0.0)
        for (e0, e1, r) in self.episodes:
            m = (t >= e0) & (t < e1)
            a, b, s = np.where(m, r, a), np.where(m, r, b), np.where(m, 0.0, s)
        label = np.where(s >= 0.5, b, a)
        return a, b, s, label

    def label_at(self, t):
        return self.mix(t)[3]

    def shift(self, t, mi):
        """Transition effect on P and R of model mi at time t: w(t) * (common + deviation of the model), w = Hann weight of the move (0 at both ends, 1 at its midpoint); 0 outside a move."""
        t = np.asarray(t, dtype=float)
        j = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.steps) - 1)
        s = np.clip((t - self.starts[j]) / np.maximum(self.ends[j] - self.starts[j], 1e-9), 0.0, 1.0)
        val = np.where(self.move[j], 0.5 * (1.0 - np.cos(2.0 * np.pi * s)) * (self.t_common[j] + self.t_dev[j, mi]), 0.0)
        for (e0, e1, r) in self.episodes:                                                           # an episode replaces the area abruptly: no mixed zone
            val = np.where((t >= e0) & (t < e1), 0.0, val)
        return val
