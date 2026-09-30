import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402
ASSETS = Path(os.environ.setdefault("EDGE_ASSETS", str(ROOT / "assets")))       # weights volume: downloaded by fetch-assets (never in the repository)
os.environ.setdefault("EDGE_DATA", str(ROOT / "data"))
for p in (ROOT, ROOT / "third_party" / "consistency"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edge_signals.base import FrameRecord  # noqa: E402


def make_boxes(n=3, conf=0.8, shift=(0.0, 0.0), seed=0):
    rng = np.random.RandomState(seed)
    b = np.zeros((n, 6))
    b[:, 0] = 0
    b[:, 1] = 0.2 + 0.6 * rng.rand(n) * 0 + np.linspace(0.2, 0.8, n) + shift[0]
    b[:, 2] = 0.5 + shift[1]
    b[:, 3:5] = 0.1
    b[:, 5] = conf
    return b


def frame(i, clip="clipA", boxes=None, feat=None, conf=0.8, backbone="bb1"):
    return FrameRecord(t=float(i), frame_id=f"{clip}-{i:07d}", clip_id=clip,
                       boxes=make_boxes(conf=conf) if boxes is None else boxes, features=feat,
                       meta={"backbone_params_id": backbone, "backbone_state_id": backbone + "-s"})


def stream(n, clip="clipA", start=0, **kw):
    return [frame(start + i, clip, **kw) for i in range(n)]


def sig_cfg(**over):
    """A minimal valid signals config (dict) with the given per-signal overrides."""
    base = {"defaults": {"window_transitions": 5, "aggregator": "median", "comparator": "previous", "rolling_windows": 2,
                         "change": "difference"},
            "signals": {"consistency": {"measure": True, "trigger": True}}}
    for k, v in over.items():
        base["signals"][k] = v
    return base


STEP_RULE = {"type": "relative_drop", "enabled": True, "stage": "screening", "comparator": "previous", "delta": 0.2, "n_consecutive": 1,
             "cooldown_windows": 2}
HOOKS = {"critical": {"enabled": False}, "cumulative": {"enabled": False}, "preventive": {"enabled": False}, "periodic": {"enabled": False}}


@pytest.fixture
def trig_cfg():
    return {"trigger": {"rules": {"step": dict(STEP_RULE), **HOOKS}}}


def asset(name: str) -> Path:
    """Path of a weights asset, or skip the test when fetch-assets has not been run."""
    p = ASSETS / "weights" / f"{name}.pt"
    if not p.exists():
        pytest.skip(f"asset {name}.pt not found under {ASSETS} (run fetch-assets)")
    return p
