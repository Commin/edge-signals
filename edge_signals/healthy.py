"""Healthy change distribution: the label-free basis of delta and of the p-value in adaptation requests.

derive_changes() runs the window statistic over healthy footage, video by video (windows never cross a video), and stores the
drop of every window against the previous window and against the anchor (the first window of the video). delta is a percentile of it.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import FrameRecord, ReferenceBundle


def derive_changes(clips: Dict[str, List[FrameRecord]], signals_cfg: dict, fps: float, signal: str = "consistency",
                   frame_stride: int = 5) -> ReferenceBundle:
    """frame_stride: every N-th frame of the source video was kept (5 at 30 fps = the 6 fps stream); recorded in the bundle, checked by `stream --video`."""
    from .config import validate_signals, validate_trigger
    from .engine import SignalEngine
    arrays, meta = {}, {}
    for comp in ("previous", "anchor"):
        cfg = {"defaults": {**signals_cfg.get("defaults", {})}, "signals": {signal: {**signals_cfg["signals"][signal], "measure": True,
                                                                                    "trigger": False, "comparator": comp}}}
        sc = validate_signals(cfg, fps)
        vals = []
        for clip, recs in clips.items():
            eng = SignalEngine(sc, validate_trigger({"trigger": {"rules": {}}}, sc), fps=fps)
            vals += [w.change for w in eng.run(recs)["windows"] if w.change is not None]
        arrays[comp] = np.array(vals, dtype=float)
        meta[f"n_windows_{comp}"] = int(len(vals))
    s = sc[signal]
    meta.update(frame_stride=frame_stride, fps=fps, signal=signal, aggregator=s["aggregator"], window_transitions=s["window_transitions"], change=s["change"], n_videos=len(clips),
                note="degradation (> 0 = worse) of every within-video window against the previous window and against the anchor; delta = the 95th percentile")
    return ReferenceBundle("healthy_changes", arrays, meta)


def delta_from(bundle: ReferenceBundle, comparator: str, percentile_of_drop: float = 95.0) -> float:
    """delta = minus the 5th percentile of the signed change = the 95th percentile of the drop."""
    return float(np.percentile(bundle.arrays[comparator], percentile_of_drop))
