"""Loading YOLO ``.txt`` prediction files into ordered per-frame box lists.

Each file holds one detection per line: ``class cx cy w h [conf]`` (normalised).
A missing or empty file for an existing frame means zero detections.

Two file-stem conventions are understood for frame ordering:

* ``{PREFIX}-{IDX:07d}``      - the index is the last hyphen-separated field (7 digits);
* ``{PREFIX}_frame{N}...``    - e.g. ``clip_frame12_jpg.rf.<hash>``.

Frames are ordered numerically within a prefix (``frame10`` after ``frame2``), and a
transition is only formed between consecutive frames of the *same* prefix.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Recognized trailing extensions when normalising a file name to a stem.
VALID_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".txt")

# The first pattern is anchored at the end, so it cannot capture the second convention
# (which ends in a hash or a bare ``_frame{N}``).
_STEM_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{7})$")
_FRAME_RE = re.compile(r"^(?P<prefix>.*?)_frame(?P<idx>\d+)")


def parse_stem(name_or_path: str) -> Optional[Tuple[str, int]]:
    """``(prefix, frame_idx)`` for either stem convention, else ``None``."""
    stem = normalize_transition_key(name_or_path, remove_final_suffix=True)
    m = _STEM_RE.match(stem) or _FRAME_RE.match(stem)
    return (m.group("prefix"), int(m.group("idx"))) if m else None


def normalize_transition_key(path_str: str, remove_final_suffix: bool = True) -> str:
    """Basename of ``path_str`` with a single recognized trailing extension stripped."""
    key = os.path.basename(str(path_str).strip())
    if remove_final_suffix:
        low = key.lower()
        for ext in VALID_EXTS:
            if low.endswith(ext):
                key = key[: -len(ext)]
                break
    return key


def extract_prefix_from_key(key: str) -> Optional[str]:
    """Return the video/sequence prefix, for either stem convention."""
    parsed = parse_stem(key)
    return parsed[0] if parsed else None


def parse_sort_key(name_or_path: str) -> Tuple[str, int, str]:
    """Numeric sort key ``(prefix, int(frame_idx), normalized_stem)``.

    Non-matching names sort last but keep a stable secondary key on the stem.
    """
    stem = normalize_transition_key(name_or_path, remove_final_suffix=True)
    parsed = parse_stem(stem)
    if parsed:
        return (parsed[0], parsed[1], stem)
    return (stem, 1 << 62, stem)


def frame_index(name_or_path: str) -> Optional[int]:
    """Integer frame index parsed from a stem/filename, or ``None``."""
    parsed = parse_stem(name_or_path)
    return parsed[1] if parsed else None


def parse_yolo_txt(filepath: str) -> List[List[float]]:
    """Parse one YOLO ``.txt`` into boxes ``[class_id(int), cx, cy, w, h, (conf)]``.

    A missing file returns ``[]`` (zero detections). Rows with fewer than 5 fields are
    skipped. A 6th field is kept as the confidence when present.
    """
    if not os.path.exists(filepath):
        return []
    boxes: List[List[float]] = []
    with open(filepath, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            box: List[float] = [
                int(float(parts[0])),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            ]
            if len(parts) >= 6:
                box.append(float(parts[5]))
            boxes.append(box)
    return boxes


def _list_label_stems(label_dir: str) -> List[str]:
    """All ``.txt`` stems in a label directory, in numeric frame order."""
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    stems = [
        normalize_transition_key(fn)
        for fn in os.listdir(label_dir)
        # macOS may create ``._*.txt`` AppleDouble metadata files when an archive
        # is copied to or extracted on FAT/exFAT media. They are binary metadata,
        # not YOLO labels, and must never enter the UTF-8 prediction parser.
        if fn.lower().endswith(".txt")
        and not fn.startswith(".")
        and os.path.isfile(os.path.join(label_dir, fn))
    ]
    return sorted(stems, key=parse_sort_key)


@dataclass
class Transition:
    """A single ``t -> t+1`` frame pair within one prefix."""

    key: str  # stem of frame t
    next_key: str  # stem of frame t+1
    prefix: str
    frame: int
    next_frame: int


@dataclass
class FrameStore:
    """Ordered per-frame detections keyed by normalized stem."""

    directory: str
    order: List[str]
    frames: Dict[str, List[List[float]]]

    def boxes(self, stem: str) -> List[List[float]]:
        return self.frames.get(stem, [])

    def transitions(self) -> List[Transition]:
        """Enumerate successive-frame transitions in numeric order.

        Transitions are formed between consecutive frames *within the same prefix*
        only. Frame indices need not differ by 1 (sampled sequences such as
        ``frame0 -> frame5`` are valid transitions).
        """
        out: List[Transition] = []
        for a, b in zip(self.order, self.order[1:]):
            pa, pb = extract_prefix_from_key(a), extract_prefix_from_key(b)
            if pa is None or pb is None or pa != pb:
                continue
            fa, fb = frame_index(a), frame_index(b)
            if fa is None or fb is None:
                continue
            out.append(Transition(key=a, next_key=b, prefix=pa, frame=fa, next_frame=fb))
        return out


def load_predictions(pred_dir: str) -> FrameStore:
    """Load every prediction ``.txt`` under ``pred_dir`` into a :class:`FrameStore`."""
    stems = _list_label_stems(pred_dir)
    frames: Dict[str, List[List[float]]] = {}
    for stem in stems:
        frames[stem] = parse_yolo_txt(os.path.join(pred_dir, stem + ".txt"))
    return FrameStore(directory=pred_dir, order=stems, frames=frames)
