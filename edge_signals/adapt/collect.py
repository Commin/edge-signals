"""collect: an AdaptationRequest (or a frame list) -> the adaptation frames + replay frames.

Adaptation frames = the frames the stream saw since the last reset (a stream time, default 0 or the last registered version's) up to the
request's time; or, without a request, an explicit frame list. Replay frames are drawn without replacement from configured lists (frames that
were already annotated for earlier training) at `ratio` x the number of adaptation frames, excluding the adaptation frames.
Writes <run>/collect/{adapt.txt, replay.txt, collect.json}.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from ..streams import order_frames
from .common import jdump, jload, read_stems, rp, sha256_file, stage_dir


def load_request(requests_path, request_id: Optional[str]) -> dict:
    rows = [json.loads(l) for l in open(requests_path) if l.strip()]
    if not rows:
        raise SystemExit(f"{requests_path}: no requests")
    if request_id is None:
        return rows[-1]
    for r in rows:
        if r.get("request_id") == request_id:
            return r
    raise SystemExit(f"request '{request_id}' not found in {requests_path}")


def frames_from_request(req: dict, stream_stems: List[str], fps: float, since_t: float) -> List[str]:
    """Frames of the stream list (ordered as the stream ordered them; t = index / fps) with since_t < t <= request t."""
    ordered = [p.stem for p in order_frames(Path(s + ".jpg") for s in stream_stems)]
    return [s for i, s in enumerate(ordered) if since_t < i / fps <= req["t"] + 1e-6]


def draw_replay(cfg_replay: dict, adapt: List[str], ratio: Optional[float] = None, seed: Optional[int] = None) -> List[str]:
    ratio = cfg_replay["ratio"] if ratio is None else ratio
    seed = cfg_replay.get("seed", 0) if seed is None else seed
    n = int(round(ratio * len(adapt)))
    if n == 0:
        return []
    pool = []
    for lst in cfg_replay["lists"]:
        pool += read_stems(lst)
    seen = set(adapt)
    pool = [s for s in pool if s not in seen]
    if n > len(pool):
        raise SystemExit(f"replay ratio {ratio} needs {n} frames but the replay lists hold {len(pool)} (after excluding the adaptation frames)")
    return random.Random(seed).sample(pool, n)


def collect(cfg: dict, run, frames_list=None, requests=None, request_id=None, stream_list=None, since_t: Optional[float] = None,
            fps: float = 6.0, ratio: Optional[float] = None, seed: Optional[int] = None, stream_out=None, mapping=None,
            since_request_id: Optional[str] = None, min_new_frames: Optional[int] = None) -> dict:
    out = stage_dir(run, "collect")
    ccfg = cfg["collect"]
    req = None
    minimum = int(ccfg.get("min_new_frames", 0) if min_new_frames is None else min_new_frames)
    if stream_out:                                              # same-video: labelled frames of the video(s) the stream saw, reset -> request
        from . import streamref as S
        videos, reqs = S.load_stream(stream_out)
        req = S.find_request(reqs, request_id)
        reset = S.position(S.find_request(reqs, since_request_id)) if since_request_id else None
        mp = S.load_mapping(rp(mapping or "data/mapping/frame_map.csv"))
        segs = S.segments(videos, reset, S.position(req))
        adapt = S.stems_in(mp, segs)
        source = {"kind": "stream_request", "stream_out": str(stream_out), "request_id": req.get("request_id"), "rule": req.get("rule"), "stage": req.get("stage"),
                  "request_position": {"video": req["video"], "frame_index": req["frame_index"]},
                  "reset_position": None if reset is None else {"video": reset[0], "frame_index": reset[1]}, "reset_request_id": since_request_id,
                  "segments": [{"video": v, "after_frame": lo, "up_to_frame": hi} for v, lo, hi in segs], "mapping": str(mapping or "data/mapping/frame_map.csv")}
        if len(adapt) < minimum:
            info = {"status": "insufficient_data", "source": source, "n_adapt": len(adapt), "min_new_frames": minimum, "n_replay": 0}
            jdump(info, out / "collect.json")
            return info
    elif frames_list:
        adapt = read_stems(frames_list)
        source = {"kind": "frame_list", "list": str(frames_list), "sha256": sha256_file(rp(frames_list))}
    else:
        if not (requests and stream_list):
            raise SystemExit("collect needs --frames-list, or --requests (+ --request-id) together with --stream-list")
        req = load_request(requests, request_id)
        if since_t is None:
            reg = jload(Path(run) / cfg.get("registry", {}).get("path", "registry.json"), {"versions": []})
            since_t = (reg["versions"][-1].get("stream_t") if reg.get("versions") else None)
            since_t = -1.0 if since_t is None else since_t              # nothing to skip: the stream start is the last reset
        adapt = frames_from_request(req, read_stems(stream_list), fps, since_t)
        source = {"kind": "request", "request_id": req.get("request_id"), "rule": req.get("rule"), "stage": req.get("stage"), "request_t": req["t"],
                  "since_t": since_t, "fps": fps, "stream_list": str(stream_list)}
    if not adapt:
        raise SystemExit("no adaptation frames collected")
    replay = draw_replay(ccfg["replay"], adapt, ratio, seed)
    (out / "adapt.txt").write_text("".join(s + "\n" for s in adapt))
    (out / "replay.txt").write_text("".join(s + "\n" for s in replay))
    info = {"status": "ok", "min_new_frames": minimum, "source": source, "n_adapt": len(adapt), "n_replay": len(replay), "replay_ratio": ccfg["replay"]["ratio"] if ratio is None else ratio,
            "replay_seed": ccfg["replay"].get("seed", 0) if seed is None else seed,
            "replay_lists": [{"list": l, "sha256": sha256_file(rp(l))} for l in ccfg["replay"]["lists"]],
            "n_clips_adapt": len({s.rsplit("-", 1)[0] for s in adapt})}
    jdump(info, out / "collect.json")
    return info
