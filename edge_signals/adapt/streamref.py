"""Same-video adaptation data: positions in a video stream <-> labelled image stems.

A stream run over videos writes requests.jsonl (each request carries the video and the ORIGINAL frame index of its last frame) and
summary.json (the videos in processing order). The frame mapping (data/mapping/frame_map.csv, metadata only) says which labelled image
stem belongs to (video, original frame index). Retraining data for a request = the labelled frames of the video(s) the stream saw between
the last reset and the request: from the reset position (exclusive) to the request position (inclusive), each video contributing its
segment of that period.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Position = Tuple[str, int]                     # (video name, original frame index)


def load_mapping(path) -> Dict[str, List[Tuple[int, str]]]:
    """{video: [(original frame index, image stem)] sorted by index}."""
    out: Dict[str, List[Tuple[int, str]]] = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["video"], []).append((int(r["frame_index"]), r["image_stem"]))
    for v in out:
        out[v].sort()
    return out


def load_stream(stream_out) -> Tuple[List[str], List[dict]]:
    """(videos in processing order, requests) of a stream run directory."""
    d = Path(stream_out)
    summ = json.loads((d / "summary.json").read_text())
    videos = summ.get("videos")
    if not videos:
        raise SystemExit(f"{d}/summary.json has no 'videos': the stream was not run on videos (use `stream --video ...`)")
    reqs = [json.loads(l) for l in open(d / "requests.jsonl") if l.strip()]
    return videos, reqs


def find_request(reqs: List[dict], request_id: Optional[str]) -> dict:
    if not reqs:
        raise SystemExit("no requests in the stream output")
    if request_id is None:
        return reqs[-1]
    for r in reqs:
        if r["request_id"] == request_id:
            return r
    raise SystemExit(f"request '{request_id}' not found")


def position(req: dict) -> Position:
    if req.get("video") is None or req.get("frame_index") is None:
        raise SystemExit(f"request {req.get('request_id')} has no video / frame_index (was the stream run on videos?)")
    return req["video"], int(req["frame_index"])


def segments(order: List[str], reset: Optional[Position], request: Position) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """[(video, lo_exclusive, hi_inclusive)] covering the stream from `reset` (exclusive; None = the stream start) to `request` (inclusive).
    A bound of None means unbounded on that side."""
    if request[0] not in order:
        raise SystemExit(f"request video {request[0]} is not in the stream's video list")
    q = order.index(request[0])
    r = order.index(reset[0]) if reset is not None else 0
    if reset is not None and reset[0] not in order:
        raise SystemExit(f"reset video {reset[0]} is not in the stream's video list")
    if (r, reset[1] if reset else -1) > (q, request[1]):
        raise SystemExit("the reset position lies after the request")
    segs = []
    for k in range(r, q + 1):
        lo = reset[1] if (reset is not None and k == r) else None
        hi = request[1] if k == q else None
        segs.append((order[k], lo, hi))
    return segs


def stems_in(mapping: Dict[str, List[Tuple[int, str]]], segs) -> List[str]:
    out = []
    for video, lo, hi in segs:
        for idx, stem in mapping.get(video, []):
            if (lo is None or idx > lo) and (hi is None or idx <= hi):
                out.append(stem)
    return out


def later_frames(mapping, request: Position) -> List[str]:
    """Labelled frames of the request's video AFTER the request (the in-stream evaluation view)."""
    return [stem for idx, stem in mapping.get(request[0], []) if idx > request[1]]
