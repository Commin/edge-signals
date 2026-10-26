"""World: areas, edge trajectories, model x area matrices, and the queries on them."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import config as cfgmod
from . import areas, frames, models, rng

FPS = frames.FPS


class World:
    def __init__(self, config=None, seed=0):
        self.seed = int(seed)
        self.cfg, self.draws = cfgmod.load(config, self.seed)
        c, sp = self.cfg, self.cfg["simple"]
        self.names = cfgmod.area_names(c)
        self.K = len(self.names)
        self.duration_s = sp["duration_s"]
        self.n_frames = int(self.duration_s * FPS)
        ap = c["area_params"]
        diff = self._per_area(ap["difficulty"], 8)
        for k, d in cfgmod.area_difficulty(c).items():             # a difficulty given in the `areas` list replaces the drawn one
            diff[k] = d
        self.difficulty = np.array(diff)
        if ap["lambda_obj"] is not None:
            lam = np.array(self._per_area(ap["lambda_obj"], 9))
        else:
            lam = ap["lambda_easy"] - (ap["lambda_easy"] - ap["lambda_hard"]) * self.difficulty
        self.lam = np.maximum(lam, np.array(self._per_area(ap["lambda_floor"], 11)))
        # edges: a scripted route, or the random timeline seen with a delay (position / front speed)
        lo, hi = c["position_range"]
        self.edges, specs = [], cfgmod.resolve_edges(c, self.names, self.duration_s)
        for i, sp_ in enumerate(specs):
            if sp_["mode"] == "route":
                pos, off = None, None
            else:
                pos = sp_["position"]
                pos = lo + (hi - lo) * float(rng.uniform(self.seed, rng.L_CONFIG, 10, i)) if pos is None else float(pos)
                off = pos / c["front_speed_kmh"] * 3600.0
            self.edges.append({"id": sp_["id"], "mode": sp_["mode"], "models": sp_["models"], "initial_model": sp_["initial_model"], "route": sp_["route"], "position_km": pos, "offset_s": off})
        self.edge_index = {e["id"]: i for i, e in enumerate(self.edges)}
        self.model_index = {m["id"]: i for i, m in enumerate(c["models"])}
        # timelines
        offs = [e["offset_s"] for e in self.edges if e["mode"] == "random"]
        tl = dict(c["timeline"])
        if tl.get("fixed_segments"):
            tl["fixed_segments"] = [[cfgmod.area_index(self.names, r, "timeline.fixed_segments"), d] for r, d in tl["fixed_segments"]]
        self.global_segments = areas.global_timeline(self.seed, self.K, tl, -max(offs) - 1.0, self.duration_s + 1.0) if offs else []
        self.ramp_s = max(1.0 / FPS, c["timeline"]["ramp_max_s"] * (1.0 - sp["steepness"]))
        ep_cfg = c["local_episodes"] if c["random_episodes"] else dict(c["local_episodes"], rate_per_hour=0)
        self.episodes, self.edge_segments, self.edge_tl = [], [], []
        for i, (e, sp_) in enumerate(zip(self.edges, specs)):
            ep = areas.local_episodes(self.seed, i, self.K, ep_cfg, self.duration_s)
            self.episodes.append(ep)
            if e["mode"] == "route":
                self.edge_segments.append(sp_["steps"])
                self.edge_tl.append(areas.RouteTimeline(sp_["steps"], ep))
            else:
                base = [(t0 + e["offset_s"], t1 + e["offset_s"], r, "global") for (t0, t1, r) in self.global_segments]
                segs = areas.overlay(base, ep)
                self.edge_segments.append(segs)
                self.edge_tl.append(areas.EdgeTimeline(segs, self.ramp_s))
        # matrices
        mt = models.build(c, self.seed, self.difficulty)
        self.P, self.R, self.F1, self.C, self.Conf = mt["P"], mt["R"], mt["F1"], mt["C"], mt["Conf"]
        self.blind, self.trained, self.warnings, self.matrix_info = mt["blind"], mt["trained"], mt["warnings"], mt

    def _per_area(self, spec, layer):
        if isinstance(spec, list):
            if len(spec) != self.K:
                raise cfgmod.ConfigError(f"expected {self.K} values (one per area), got {len(spec)}")
            return [float(x) for x in spec]
        if isinstance(spec, dict):
            return [float(spec["range"][0] + (spec["range"][1] - spec["range"][0]) * float(rng.uniform(self.seed, rng.L_CONFIG, layer, k))) for k in range(self.K)]
        return [float(spec)] * self.K

    # ------------------------------------------------------------------ queries
    def _e(self, e):
        if isinstance(e, str):
            if e not in self.edge_index:
                raise KeyError(f"unknown edge '{e}' (edges: {', '.join(self.edge_index)})")
            return self.edge_index[e]
        return int(e)

    def _m(self, m):
        if isinstance(m, str):
            if m not in self.model_index:
                raise KeyError(f"unknown model '{m}' (models: {', '.join(self.model_index)})")
            return self.model_index[m]
        return int(m)

    def edge_models(self, e):
        """Ids of the models stored on edge e (any model of the world can still be run on it: it is fetched from the edge that stores it)."""
        return list(self.edges[self._e(e)]["models"])

    def model_source(self, e, m, from_edge=None):
        """-> ("local" | "fetched", id of the edge the model comes from). A model that no edge stores is "fetched" from None."""
        eid = self.edges[self._e(e)]["id"]
        if from_edge is not None:
            fe = self.edges[self._e(from_edge)]
            if m not in fe["models"]:
                raise ValueError(f"model '{m}' is not stored on edge '{fe['id']}'")
            return ("local", eid) if fe["id"] == eid else ("fetched", fe["id"])
        if m in self.edges[self._e(e)]["models"]:
            return "local", eid
        owners = [x["id"] for x in self.edges if m in x["models"]]
        return "fetched", (owners[0] if owners else None)

    def area(self, e, t):
        """Name of the area of edge e at time t (seconds; scalar or array). During a move: the area that is nearer (the one with the larger weight)."""
        idx = self.edge_tl[self._e(e)].label_at(t)
        return self.names[int(idx)] if np.ndim(idx) == 0 else np.array(self.names)[idx]

    def quality(self, e, m, t):
        """P, R and F1 of model m on the data of edge e at time t (arrays): P and R are interpolated linearly during a move and F1 is recomputed from them."""
        a, b, s, _ = self.edge_tl[self._e(e)].mix(t)
        mi = self._m(m)
        P, R = (1 - s) * self.P[mi, a] + s * self.P[mi, b], (1 - s) * self.R[mi, a] + s * self.R[mi, b]
        return {"P": P, "R": R, "F1": models.f1_of(P, R)}

    def segments(self, e):
        i = self._e(e)
        if self.edges[i]["mode"] == "route":
            return [{"t0": s["t0"], "t1": s["t1"], "kind": s["kind"], "area": self.names[s["area"]], **({"from_area": self.names[s["from_area"]]} if s["kind"] == "move" else {}), "origin": "route"}
                    for s in self.edge_segments[i]]
        out = []
        for (t0, t1, r, o) in self.edge_segments[i]:
            if t1 > 0 and t0 < self.duration_s:
                out.append({"t0": max(t0, 0.0), "t1": min(t1, self.duration_s), "area": self.names[int(r)], "origin": o})
        return out

    def frame_truth(self, e, m, f0, f1):
        a = frames.frame_arrays(self, self._e(e), self._m(m), int(f0), int(f1))
        return {k: a[k] for k in ("frame", "n_obj", "TP", "FP", "FN")}

    def clip_score(self, e, m, t_end, n_pairs=30):
        """The clip is always the data of edge e (the requester), whatever node hosts model m: the n_pairs frame pairs that end at t_end."""
        f_end = int(math.floor(t_end * FPS + 1e-9))
        f_start = max(f_end - n_pairs, 0)
        a = frames.frame_arrays(self, self._e(e), self._m(m), f_start, f_end + 1)
        cons, cv = a["cons"], a["cons_valid"]
        conf = a["conf"][1:][a["conf_valid"][1:]]
        return {"consistency": float(np.median(cons[cv])) if cv.any() else None, "confidence": float(np.median(conf)) if len(conf) else None,
                "valid_fraction": float(cv.mean()) if len(cv) else 0.0, "n_pairs": int(len(cons)), "areas": [self.names[k] for k in sorted({int(x) for x in a["area"][1:]})]}

    def stream(self, e, schedule, out_dir, log_signals=True, delta=None, logs="full"):
        """logs: "full" (signals, windows, requests, context, frames, truth) or "minimal" (windows, requests, context, truth)."""
        from . import stream as st
        return st.run(self, self._e(e), schedule, out_dir, log_signals, delta, logs)

    # ------------------------------------------------------------------ ground truth
    def to_dict(self):
        mid = [m["id"] for m in self.cfg["models"]]
        mat = lambda X: {mid[i]: [round(float(x), 5) for x in X[i]] for i in range(len(mid))}
        r5 = lambda x: round(float(x), 5)
        return {"synthetic": True, "seed": self.seed, "fps": FPS, "window_pairs": 30, "duration_s": self.duration_s, "n_areas": self.K, "config": self.cfg, "range_draws": self.draws,
                "areas": [{"name": n, "difficulty": r5(self.difficulty[k]), "lambda_obj": r5(self.lam[k])} for k, n in enumerate(self.names)],
                "edges": self.edges, "ramp_s": self.ramp_s,
                "global_timeline": [{"t0": a, "t1": b, "area": self.names[int(r)]} for (a, b, r) in self.global_segments],
                "edge_segments": {e["id"]: self.segments(e["id"]) for e in self.edges},
                "local_episodes": {e["id"]: [{"t0": a, "t1": b, "area": self.names[int(r)]} for (a, b, r) in self.episodes[i]] for i, e in enumerate(self.edges)},
                "models": self.cfg["models"], "precision": mat(self.P), "recall": mat(self.R), "f1": mat(self.F1), "consistency_centre": mat(self.C), "confidence_centre": mat(self.Conf),
                "blind_cells": [{"model": mid[i], "area": self.names[int(k)]} for i, k in zip(*np.nonzero(self.blind))], "trained": mat(self.trained.astype(float)),
                "intercepts": {"consistency": self.matrix_info["a_c"], "confidence": self.matrix_info["a_f"]},
                "warnings": self.warnings}

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=float))
        return p
