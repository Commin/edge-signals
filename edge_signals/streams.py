"""Helpers to feed the engine from image files: clip ids from file names, chunked inference, reference fitting."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .base import FrameRecord

IMG_EXT = (".jpg", ".jpeg", ".png")
_STEM = re.compile(r"^(?P<clip>.+)-(?P<idx>\d+)$")


def clip_and_index(stem: str):
    """`<clip>-<frame index>` -> (clip, int index); other names -> (None, None)."""
    m = _STEM.match(stem)
    return (m.group("clip"), int(m.group("idx"))) if m else (None, None)


def read_list(path, root=None) -> List[Path]:
    """One image per line, either a full path or a stem (then <root>/<stem>.jpg is used)."""
    root = Path(root) if root else None
    out = []
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        p = Path(s)
        if p.suffix.lower() not in IMG_EXT:
            p = (root / f"{s}.jpg") if root else Path(f"{s}.jpg")
        elif root and not p.is_absolute():
            p = root / p
        out.append(p)
    return out


def order_frames(paths: Iterable[Path]) -> List[Path]:
    """Numeric order within a clip; clips in name order."""
    def key(p):
        c, i = clip_and_index(p.stem)
        return (c or p.stem, i if i is not None else 0)
    return sorted(paths, key=key)


def stream_records(detector, paths: List[Path], fps: float, chunk: int = 64, device_id: str = "device",
                   model_version: str = "model", clip_from_name: bool = True) -> Iterator[FrameRecord]:
    """Run the detector chunk by chunk and yield FrameRecords in stream time (t = index / fps)."""
    n = 0
    for i in range(0, len(paths), chunk):
        part = paths[i:i + chunk]
        clips = [clip_and_index(p.stem)[0] if clip_from_name else None for p in part]
        recs = detector.infer([str(p) for p in part], [p.stem for p in part], t0=n / fps, fps=fps, clip_ids=clips,
                              device_id=device_id, model_version=model_version)
        n += len(part)
        yield from recs
