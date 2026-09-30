"""Plugin interface of the signal framework: records, SignalValue, Signal, ReferenceBundle.

A signal is one module under edge_signals/builtin/. It declares what it is (name, family, granularity,
requires, higher_is_healthier, label_free) and implements update(record) -> SignalValue. An invalid value is
NEVER replaced by a number: value is None, valid is False and invalid_reason says why.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FAMILIES = ("quality", "drift", "context")
GRANULARITIES = ("frame", "pair", "window")
REQUIRES = ("preds", "features", "image")


@dataclass
class FrameRecord:
    """One frame as produced by the edge inference call (one forward pass)."""
    t: float                                   # stream time in seconds
    frame_id: str
    boxes: np.ndarray                          # (n, 6) [cls, cx, cy, w, h, conf], normalised coordinates
    clip_id: Optional[str] = None              # replay only: the clip (video) the frame belongs to
    device_id: str = "device"
    model_version: str = "model"
    features: Optional[np.ndarray] = None      # (C,) globally pooled backbone vector, same forward pass
    image: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)   # e.g. {"backbone_id": "..."}
    video: Optional[str] = None                # video input: the video name ...
    frame_index: Optional[int] = None          # ... and the ORIGINAL frame index (0-based decode order) of this frame


@dataclass
class PairRecord:
    """Two consecutive frames of the same clip (prev -> cur)."""
    prev: FrameRecord
    cur: FrameRecord
    cache: Dict[str, Any] = field(default_factory=dict)   # results shared between signals for this pair

    @property
    def t(self):
        return self.cur.t


@dataclass
class WindowRecord:
    """The frames of one closed window (for granularity == 'window' signals)."""
    frames: List[FrameRecord]
    clip_id: Optional[str] = None

    @property
    def t(self):
        return self.frames[-1].t


@dataclass
class SignalValue:
    signal: str
    version: str
    value: Optional[float]
    valid: bool
    invalid_reason: Optional[str]
    t: float
    device_id: str
    model_version: str
    reference_id: Optional[str]
    granularity: str = "frame"
    frame_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("signal", "version", "value", "valid", "invalid_reason", "t", "device_id",
                                            "model_version", "reference_id", "granularity", "frame_id")}
        d["extras"] = {k: clean(v) for k, v in self.extras.items()}
        d["value"] = clean(d["value"])
        return d


def clean(v):
    """JSON-safe number: NaN / inf -> None (an invalid number is never written as a number)."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class ReferenceBundle:
    """Arrays + JSON metadata. reference_id = sha256 of the content (arrays in sorted key order + metadata)."""

    def __init__(self, signal: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]):
        self.signal, self.arrays, self.meta = signal, arrays, meta
        h = hashlib.sha256()
        for k in sorted(arrays):
            h.update(k.encode())
            h.update(np.ascontiguousarray(arrays[k]).tobytes())
        h.update(json.dumps(meta, sort_keys=True, default=str).encode())
        self.id = h.hexdigest()[:16]

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.signal}.{self.id}.npz"
        np.savez(path, __signal__=np.array(self.signal), __meta__=np.array(json.dumps(self.meta, sort_keys=True, default=str)),
                 **self.arrays)
        (directory / f"{self.signal}.{self.id}.json").write_text(json.dumps({"reference_id": self.id, "signal": self.signal,
                                                                             **self.meta}, indent=2, sort_keys=True, default=str))
        return path

    @classmethod
    def load(cls, path) -> "ReferenceBundle":
        z = np.load(path, allow_pickle=False)
        arrays = {k: z[k] for k in z.files if not k.startswith("__")}
        b = cls(str(z["__signal__"]), arrays, json.loads(str(z["__meta__"])))
        stem_id = Path(path).stem.rsplit(".", 1)[-1]
        if stem_id != b.id:
            raise ValueError(f"reference bundle {path} is corrupt: file id {stem_id} != content hash {b.id}")
        return b


class Signal:
    """Base class. Subclasses set the class attributes and implement update()."""
    name: str = ""
    version: str = "1"
    family: str = ""                    # quality | drift | context
    granularity: str = ""               # frame | pair | window
    requires: frozenset = frozenset()   # subset of {"preds", "features", "image"}
    higher_is_healthier: bool = True    # for context signals: False means "larger = more disturbance"
    label_free: bool = True

    def __init__(self, params: Optional[dict] = None, reference: Optional[ReferenceBundle] = None):
        self.params = dict(params or {})
        self.reference = reference

    @property
    def reference_id(self) -> Optional[str]:
        return self.reference.id if self.reference is not None else None

    def value(self, record, value, extras=None) -> SignalValue:
        v = None if value is None or not math.isfinite(float(value)) else float(value)
        return self._make(record, v, v is not None, None if v is not None else "non_finite", extras)

    def invalid(self, record, reason: str, extras=None) -> SignalValue:
        return self._make(record, None, False, reason, extras)

    def _make(self, record, value, valid, reason, extras) -> SignalValue:
        cur = record.cur if isinstance(record, PairRecord) else record
        if isinstance(record, WindowRecord):
            cur = record.frames[-1]
        return SignalValue(signal=self.name, version=self.version, value=value, valid=valid, invalid_reason=reason,
                           t=record.t, device_id=cur.device_id, model_version=cur.model_version,
                           reference_id=self.reference_id, granularity=self.granularity, frame_id=cur.frame_id,
                           extras=dict(extras or {}))

    def update(self, record) -> SignalValue:            # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def fit_reference(cls, records, params: Optional[dict] = None) -> Optional[ReferenceBundle]:
        """Fit a reference bundle from frame records (default: the signal needs none)."""
        return None

    def reference_level(self, aggregator: str, window: int) -> Optional[float]:
        """Expected window statistic under the reference (used by comparator 'reference'); None if not available."""
        return None
