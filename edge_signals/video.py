"""Video input for the stream stage: decode with OpenCV, keep every `frame_stride`-th frame (default 5: 30 fps -> 6 fps).

Every processed frame keeps its video name and its ORIGINAL frame index (0-based decode order), which is what the frame mapping and
the same-video retraining use. The stride is tied to the calibration: delta, the healthy change distribution and the window length were
derived on a 6 fps stream (every 5th frame of a 30 fps video), so another stride is refused unless the configuration points to a
healthy change distribution calibrated at that stride.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .base import FrameRecord

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")


def video_name(path) -> str:
    return Path(path).stem


def read_video_list(path) -> List[Path]:
    """One video per line (a path); blank lines and # comments ignored."""
    out = []
    for line in open(path):
        s = line.split("#")[0].strip()
        if s:
            out.append(Path(s))
    return out


def probe(path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    info = {"fps": float(cap.get(cv2.CAP_PROP_FPS)), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def iter_frames(path, stride: int = 5, first: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
    """(original frame index, BGR frame) for every index with index % stride == 0 (counted from 0); other frames are only grabbed, not decoded to arrays."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    i = 0
    try:
        while True:
            if i % stride == 0 and i >= first:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
            else:
                if not cap.grab():
                    break
            i += 1
    finally:
        cap.release()


def check_stride(stride: int, effective_fps_cfg: float, source_fps: float, calibrated_stride: Optional[int], calibration_note: str = "") -> float:
    """Refuse a stride the thresholds were not calibrated for; return the effective fps of the processed stream."""
    if stride < 1:
        raise SystemExit("frame_stride must be >= 1")
    if calibrated_stride is None:
        raise SystemExit("the healthy change distribution does not record the frame stride it was calibrated at; refusing to run on video "
                         "(re-derive it with edge_signals.healthy.derive_changes(..., frame_stride=...))")
    if stride != calibrated_stride:
        raise SystemExit(f"frame_stride {stride} differs from the stride the calibration was made at ({calibrated_stride}): delta, the healthy change distribution and "
                         f"the window length are only valid at that stride. Point trigger.healthy_changes at a distribution calibrated at stride {stride} "
                         f"(its meta frame_stride must be {stride}) and re-derive delta before running with this stride. {calibration_note}")
    eff = source_fps / stride
    if abs(eff - effective_fps_cfg) > 0.25:
        raise SystemExit(f"a {source_fps:.2f} fps video at frame_stride {stride} gives {eff:.2f} fps, but the calibration (stream.fps) is {effective_fps_cfg:.2f} fps")
    return eff


def stream_video_records(detector, videos: Sequence, stride: int, fps: float, chunk: int, device_id: str = "device", model_version: str = "model"
                         ) -> Iterator[FrameRecord]:
    """Run the detector chunk by chunk over the decoded frames of the videos (in the given order); yield FrameRecords.
    t = running index / fps over the whole stream; clip_id = video name; every record carries video and the ORIGINAL frame index."""
    n = 0
    buf: list = []                                   # (video, index, frame)

    def flush():
        nonlocal n
        recs = detector.infer([f for _, _, f in buf], [f"{v}@{i:07d}" for v, i, _ in buf], t0=n / fps, fps=fps, clip_ids=[v for v, _, _ in buf],
                              device_id=device_id, model_version=model_version)
        for r, (v, i, _) in zip(recs, buf):
            r.video, r.frame_index = v, i
        n += len(buf)
        out = list(recs)
        buf.clear()
        return out

    for path in videos:
        name = video_name(path)
        for idx, frame in iter_frames(path, stride):
            buf.append((name, idx, frame))
            if len(buf) >= chunk:
                yield from flush()
    if buf:
        yield from flush()
