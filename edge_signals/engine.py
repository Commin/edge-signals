"""SignalEngine: turns a stream of FrameRecords into SignalValues, window statistics, trigger events and context events.

* Windows never cross a clip change; the comparison state (previous / anchor / rolling) does (one continuous stream).
* Every clip change is logged as a context event (scene_cut). No pair is formed across a clip change.
* A frame-granularity value enters its window when the frame has served as the first frame of a pair (so a frame
  window covers the same frames as the pair window of the same size).
* Outputs are JSONL files in the (configurable) output directory: signals, windows, requests (adaptation requests), context.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import registry
from .base import FrameRecord, PairRecord, ReferenceBundle, WindowRecord
from .trigger import TriggerEngine
from .windows import WindowTracker


class SignalEngine:
    def __init__(self, signals_cfg: Dict[str, dict], trigger_cfg: dict, references: Optional[Dict[str, ReferenceBundle]] = None,
                 out_dir=None, log_values: bool = True, fps: Optional[float] = None,
                 healthy: Optional[ReferenceBundle] = None):
        self.cfg = signals_cfg
        self.signals = {}
        self.trackers = {}
        refs = references or {}
        for name, s in signals_cfg.items():
            if not s["measure"]:
                continue
            cls = registry.get(name)
            self.signals[name] = cls(s.get("params"), refs.get(name))
            self.trackers[name] = WindowTracker(self.signals[name], s, fps)
        self.trigger = TriggerEngine(trigger_cfg, [n for n, s in signals_cfg.items() if s["trigger"] and s["measure"]], healthy)
        self.trigger.register(self.trackers)
        self.prev: Optional[FrameRecord] = None
        self._prev_vals: dict = {}
        self.out_dir = Path(out_dir) if out_dir else None
        self._fh = {}
        self.log_values = log_values
        self.timing = {n: 0.0 for n in self.signals}
        self.n_frames = self.n_pairs = 0
        self.n_requests = 0
        self.latest_window: dict = {}
        self.latest_value: dict = {}
        self.dropped_partial_windows = 0
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for k in ("signals", "windows", "requests", "context", "frames"):
                self._fh[k] = open(self.out_dir / f"{k}.jsonl", "w")

    # ------------------------------------------------------------------ logging
    def _log(self, kind, d):
        if kind in self._fh and (self.log_values or kind != "signals"):
            self._fh[kind].write(json.dumps(d) + "\n")

    def close(self):
        for fh in self._fh.values():
            fh.close()

    # ------------------------------------------------------------------ processing
    def process(self, fr: FrameRecord) -> dict:
        """Process one frame. Returns {"values": [...], "windows": [...], "triggers": [...], "context": [...]}."""
        out = {"values": [], "windows": [], "triggers": [], "context": []}
        self.n_frames += 1
        self._log("frames", {"frame_id": fr.frame_id, "t": fr.t, "clip_id": fr.clip_id, "video": fr.video, "frame_index": fr.frame_index,
                             "n_boxes": int(len(fr.boxes))})
        prev = self.prev
        if prev is not None and prev.clip_id != fr.clip_id:
            ev = {"kind": "scene_cut", "t": fr.t, "from_clip": prev.clip_id, "to_clip": fr.clip_id, "frame_id": fr.frame_id,
                  "device_id": fr.device_id}
            out["context"].append(ev)
            self._log("context", ev)
            for tr in self.trackers.values():
                tr.close_clip()
            prev = None
        frame_vals = {}
        for name, sig in self.signals.items():
            if sig.granularity == "frame":
                t0 = time.perf_counter()
                frame_vals[name] = sig.update(fr)
                self.timing[name] += time.perf_counter() - t0
        for sv in frame_vals.values():
            self.latest_value[sv.signal] = sv
            out["values"].append(sv)
            self._log("signals", sv.to_dict())
        if prev is not None:
            self.n_pairs += 1
            pair = PairRecord(prev, fr)
            for name, sig in self.signals.items():
                if sig.granularity == "pair":
                    t0 = time.perf_counter()
                    sv = sig.update(pair)
                    self.timing[name] += time.perf_counter() - t0
                    out["values"].append(sv)
                    self.latest_value[name] = sv
                    self._log("signals", sv.to_dict())
                    self._push(name, sv, fr.clip_id, out)
                elif sig.granularity == "frame":
                    self._push(name, self._prev_vals[name], prev.clip_id, out)   # prev frame served as the first frame of a pair
        self._prev_vals = frame_vals
        self.prev = fr
        for f in out.pop("_fired", []):          # requests are built after all signals of this frame are processed: the context is current
            req = self.trigger.request(f, self._context(f.signal), self.n_requests, (fr.video, fr.frame_index))
            self.n_requests += 1
            out["triggers"].append(req)
            self._log("requests", req.to_dict())
        return out

    def _context(self, exclude: str) -> dict:
        """Latest values of every measured signal except the firing one - context only, never read by a rule."""
        ctx = {}
        for name, sig in self.signals.items():
            if name == exclude:
                continue
            ws, lv = self.latest_window.get(name), self.latest_value.get(name)
            ctx[name] = {"window_statistic": None if ws is None else ws.statistic, "window_id": None if ws is None else ws.window_id,
                         "aggregator": None if ws is None else ws.aggregator, "window_valid": None if ws is None else ws.valid,
                         "latest_value": None if lv is None else lv.value, "latest_valid": None if lv is None else lv.valid,
                         "latest_invalid_reason": None if lv is None else lv.invalid_reason}
        return ctx

    def _push(self, name, sv, clip_id, out):
        tr = self.trackers[name]
        ws = tr.push(sv, clip_id)
        if ws is None:
            return
        out["windows"].append(ws)
        self._log("windows", dict(ws.to_dict(), kind="window"))
        self.latest_window[name] = ws
        out.setdefault("_fired", []).extend(self.trigger.on_window(ws))

    def run(self, frames: Iterable[FrameRecord]) -> dict:
        allout = {"values": [], "windows": [], "triggers": [], "context": []}
        for fr in frames:
            o = self.process(fr)
            for k in allout:
                allout[k] += o[k]
        self.dropped_partial_windows = sum(t.dropped_partial for t in self.trackers.values())
        self.close()
        return allout
