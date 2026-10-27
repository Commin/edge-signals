"""A model schedule (input) -> the package's real SignalEngine / WindowTracker / TriggerEngine, plus truth.jsonl.

The schedule is data, not a decision: this module contains no model selection. This module contains no window or trigger logic either: windows,
comparison, p-value, robust_z and the requests are the package's own code. A model change is a scene cut: no pair across it, the partial window is dropped,
the trigger state is reset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import frames

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "third_party" / "consistency"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FPS = frames.FPS
CHUNK = 20000
STATUS = {0: "MATCHED", 1: "ONE_SIDED", 2: "EMPTY_PAIR"}
REASON = {"consistency": "EMPTY_PAIR", "confidence": "no_boxes"}


def parse_schedule(sched, model_ids, default=None):
    """[(t, model_id, from_node or None)] sorted; the first entry must be at t = 0. An entry is (t, model), (t, model, from_node) or {"t", "model", "from_node"}."""
    if not sched:
        return [(0.0, default or model_ids[0], None)]
    out = []
    for e in sched:
        if isinstance(e, dict):
            t, m, fe = e["t"], e["model"], e.get("from_node")
        else:
            t, m, fe = e[0], e[1], (e[2] if len(e) > 2 else None)
        if m not in model_ids:
            raise ValueError(f"schedule: unknown model '{m}'")
        out.append((float(t), m, fe))
    out.sort(key=lambda x: x[0])
    if out[0][0] != 0.0:
        raise ValueError("schedule: the first entry must be at t = 0")
    return out


def parse_schedule_text(text, model_ids):
    """'0:s_a,3600:x_b,7200:x_c@cloud': time in seconds, model id, optionally @node the model is fetched from (an edge or a cloud node)."""
    ent = []
    for p in text.split(","):
        if not p.strip():
            continue
        a, b = p.split(":", 1)
        m, _, fe = b.partition("@")
        ent.append((float(a), m.strip(), fe.strip() or None))
    return parse_schedule(ent, model_ids)


def make_engine(fps, out_dir=None, log_signals=True, delta=None, minimal=False):
    from edge_signals.base import FrameRecord, PairRecord, ReferenceBundle
    from edge_signals.config import load_configs
    from edge_signals.engine import SignalEngine
    sc, tc = load_configs(ROOT / "configs" / "signals.yaml", ROOT / "configs" / "trigger.yaml", fps)
    for name in sc:
        sc[name]["measure"] = name in ("consistency", "confidence")
        sc[name]["trigger"] = name == "consistency"
    if delta is not None:
        tc["rules"]["step"]["delta"] = float(delta)
    healthy = ReferenceBundle.load(ROOT / tc["healthy_changes"]) if tc.get("healthy_changes") else None

    class SyntheticEngine(SignalEngine):
        """SignalEngine fed with synthetic values instead of detections (the same orchestration as SignalEngine.process, with the logging marked synthetic)."""

        def _log(self, kind, d):
            super()._log(kind, dict(d, synthetic=True))

        def reset_trigger_state(self):
            for tr in self.trackers.values():
                tr.reset_reference()
            for r in self.trigger.rules:
                for st in getattr(r, "state", {}).values():
                    st.reset()
            for st in self.trigger.ev_state.values():
                st.reset()
            self.latest_window, self.latest_value = {}, {}

        def feed(self, fr, nbox, pair_vals, frame_vals):
            """pair_vals / frame_vals: {signal: (value or None, invalid_reason, extras)}."""
            out = {"windows": [], "triggers": [], "context": [], "values": []}
            self.n_frames += 1
            self._log("frames", {"frame_id": fr.frame_id, "t": fr.t, "clip_id": fr.clip_id, "video": None, "frame_index": None, "n_boxes": int(nbox)})
            prev = self.prev
            if prev is not None and prev.clip_id != fr.clip_id:                 # a model change = a scene cut
                ev = {"kind": "scene_cut", "t": fr.t, "from_clip": prev.clip_id, "to_clip": fr.clip_id, "frame_id": fr.frame_id, "device_id": fr.device_id}
                out["context"].append(ev)
                self._log("context", ev)
                for tr in self.trackers.values():
                    tr.close_clip()                                                # the partial window is dropped
                self.reset_trigger_state()
                prev = None
            fvals = {}
            for name, sig in self.signals.items():
                if sig.granularity == "frame":
                    v, why, ex = frame_vals[name]
                    fvals[name] = sig.invalid(fr, why, ex) if v is None else sig.value(fr, v, ex)
            for sv in fvals.values():
                self.latest_value[sv.signal] = sv
                self._log("signals", sv.to_dict())
            if prev is not None:
                self.n_pairs += 1
                pair = PairRecord(prev, fr)
                for name, sig in self.signals.items():
                    if sig.granularity == "pair":
                        v, why, ex = pair_vals[name]
                        sv = sig.invalid(pair, why, ex) if v is None else sig.value(pair, v, ex)
                        self.latest_value[name] = sv
                        self._log("signals", sv.to_dict())
                        self._push(name, sv, fr.clip_id, out)
                    elif sig.granularity == "frame":
                        self._push(name, self._prev_vals[name], prev.clip_id, out)
            self._prev_vals = fvals
            self.prev = fr
            for f in out.pop("_fired", []):
                req = self.trigger.request(f, self._context(f.signal), self.n_requests, (None, None))      # image-style: never mappable to labelled frames
                self.n_requests += 1
                out["triggers"].append(req)
                self._log("requests", req.to_dict())
            return out

    eng = SyntheticEngine(sc, tc, {}, out_dir=out_dir, log_values=log_signals, fps=fps, healthy=healthy)
    if minimal and out_dir:                                      # --logs minimal: windows, requests, context (and truth) only
        for k in ("signals", "frames"):
            eng._fh.pop(k).close()
            (Path(out_dir) / f"{k}.jsonl").unlink()
    return eng, FrameRecord


def run(w, ei, schedule, out_dir=None, log_signals=True, delta=None, logs="full"):
    """Feed the schedule through the real engine. With out_dir: signals / windows / requests / context / frames.jsonl, truth.jsonl, summary.json. Returns the engine results."""
    edge = w.edges[ei]["id"]
    mids = [m["id"] for m in w.cfg["models"]]
    sched = parse_schedule(schedule, mids)
    src = [w.model_source(ei, m, fe) for _, m, fe in sched]                 # any model can be run: a model that is not stored on the edge is fetched from the node that stores it
    N = w.n_frames
    fb = [min(int(round(t * FPS)), N) for t, _, _ in sched] + [N]
    eng, FrameRecord = make_engine(FPS, out_dir, log_signals, delta, minimal=(logs == "minimal"))
    truth = open(Path(out_dir) / "truth.jsonl", "w") if out_dir else None
    res = {"windows": [], "triggers": [], "context": []}
    for si, (t0, m, _fe) in enumerate(sched):
        mi = w.model_index[m]
        seg0, seg1 = fb[si], fb[si + 1]
        c0 = seg0
        while c0 < seg1:
            c1 = min(c0 + CHUNK, seg1)
            start = c0 - 1 if c0 > seg0 else c0                                  # overlap one frame so that the pair (c0-1, c0) exists inside a segment
            a = frames.frame_arrays(w, ei, mi, start, c1)
            for p in range(c0 - start if start < c0 else 0, c1 - start):
                f = start + p
                fr = FrameRecord(t=f / FPS, frame_id=f"synthetic-{edge}-{f:07d}", boxes=np.zeros((0, 6)), clip_id=m, device_id=f"synthetic-{edge}", model_version=f"synthetic-{m}")
                nb = int(a["nbox"][p])
                fv = {"confidence": (float(a["conf"][p]) if a["conf_valid"][p] else None, REASON["confidence"], {"n_boxes": nb})}
                pv = {}
                if p >= 1 and f > seg0:
                    q = p - 1
                    st = int(a["cons_status"][q])
                    n0, n1 = int(a["nbox"][p - 1]), nb
                    ex = {"alarm": None, "status": STATUS[st], "G": None, "G_mc": None, "dx": None, "dy": None, "K": 0 if st else None, "N_t": n0, "N_t1": n1}
                    pv = {"consistency": (float(a["cons"][q]) if a["cons_valid"][q] else None, REASON["consistency"], ex)}
                o = eng.feed(fr, nb, pv, fv)
                res["windows"] += o["windows"]
                res["triggers"] += o["triggers"]
                res["context"] += o["context"]
                if truth is not None:
                    truth.write(json.dumps({"synthetic": True, "frame_index": f, "t": f / FPS, "model_id": m, "area": w.names[int(a["area"][p])], "n_obj": int(a["n_obj"][p]),
                                            "TP": int(a["TP"][p]), "FP": int(a["FP"][p]), "FN": int(a["FN"][p])}) + "\n")
            c0 = c1
    eng.dropped_partial_windows = sum(t.dropped_partial for t in eng.trackers.values())
    eng.close()
    if truth is not None:
        truth.close()
        wins = res["windows"]
        summary = {"synthetic": True, "edge": edge, "seed": w.seed, "model_schedule": [{"t": t, "model": m, "source": sc, "from_node": fe} for (t, m, _), (sc, fe) in zip(sched, src)], "n_frames": eng.n_frames, "n_pairs": eng.n_pairs,
                   "n_windows": len(wins), "n_windows_by_signal": {n: sum(1 for x in wins if x.signal == n) for n in sorted({x.signal for x in wins})}, "n_requests": len(res["triggers"]),
                   "dropped_partial_windows": eng.dropped_partial_windows, "delta": eng.trigger.flag_rule.delta,
                   "n_windows_note": "n_windows counts window records over both signals (as in windows.jsonl); n_windows_by_signal gives the count per signal"}
        (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    res["engine"], res["schedule"] = eng, [(t, m) for t, m, _ in sched]
    res["sources"] = src
    return res
