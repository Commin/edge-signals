"""Replay order and video resolution for `stream --videos-dir`.

configs/replay_order.yaml says WHICH original videos are replayed and in WHICH order (video codes, e.g. MVI_0788_VIS_OB). The user points
`--videos-dir` at any folder holding their own copy of the videos; every code is looked up there:

  1. exact match of the file stem with the code (case-insensitive);
  2. otherwise a file stem that CONTAINS the code;
  3. more than one candidate remains: ambiguous -> reported with its candidates and treated as missing (never picked silently).

The folder is searched recursively; .avi .mp4 .mov .mkv (case-insensitive) are accepted. Nothing is downloaded, nothing is copied.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv")


def load_order(path) -> dict:
    o = yaml.safe_load(open(path))
    for k in ("splits", "streams", "default_stream"):
        if k not in o:
            raise SystemExit(f"{path}: missing '{k}'")
    for s, codes in o["splits"].items():
        if len(set(codes)) != len(codes):
            raise SystemExit(f"{path}: duplicate video code in split {s}")
    for n, sp in o["streams"].items():
        for s in sp:
            if s not in o["splits"]:
                raise SystemExit(f"{path}: stream {n} names unknown split {s}")
    return o


def ordered_codes(order: dict, stream: Optional[str] = None, split: Optional[str] = None) -> List[dict]:
    """[{code, split}] in replay order for a named stream (default stream if neither is given) or a single split."""
    if stream and split:
        raise SystemExit("give --stream or --split, not both")
    if split:
        if split not in order["splits"]:
            raise SystemExit(f"unknown split '{split}' (available: {', '.join(order['splits'])})")
        parts = [split]
        name = f"split:{split}"
    else:
        name = stream or order["default_stream"]
        if name not in order["streams"]:
            raise SystemExit(f"unknown stream '{name}' (available: {', '.join(order['streams'])})")
        parts = order["streams"][name]
    out, seen = [], set()
    for s in parts:
        for c in order["splits"][s]:
            if c in seen:
                raise SystemExit(f"video {c} appears twice in {name}")
            seen.add(c)
            out.append({"code": c, "split": s})
    return out


def find_videos(videos_dir) -> List[Path]:
    d = Path(videos_dir)
    if not d.is_dir():
        raise SystemExit(f"--videos-dir is not a directory: {d}")
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.name.startswith("."))


def resolve_code(code: str, files: Sequence[Path]) -> dict:
    c = code.lower()
    exact = [p for p in files if p.stem.lower() == c]
    if len(exact) == 1:
        return {"status": "exact", "file": str(exact[0])}
    if len(exact) > 1:
        return {"status": "ambiguous", "match": "exact", "candidates": [str(p) for p in exact]}
    part = [p for p in files if c in p.stem.lower()]
    if len(part) == 1:
        return {"status": "contains", "file": str(part[0])}
    if len(part) > 1:
        return {"status": "ambiguous", "match": "contains", "candidates": [str(p) for p in part]}
    return {"status": "missing"}


def resolve(order_items: Sequence[dict], videos_dir, on_missing: str = "skip") -> dict:
    """Resolve every code. Returns {videos_dir, on_missing, resolution: {code: {...}}, replayed: [{code, split, file}], not_replayed: [...]}."""
    if on_missing not in ("skip", "error"):
        raise SystemExit("--on-missing must be skip or error")
    files = find_videos(videos_dir)
    res, replayed, skipped = {}, [], []
    for it in order_items:
        r = resolve_code(it["code"], files)
        r["split"] = it["split"]
        res[it["code"]] = r
        if r["status"] in ("exact", "contains"):
            replayed.append({"code": it["code"], "split": it["split"], "file": r["file"]})
        else:
            skipped.append({"code": it["code"], "split": it["split"], "reason": r["status"], **({"candidates": r["candidates"]} if "candidates" in r else {})})
    return {"videos_dir": str(videos_dir), "n_video_files_found": len(files), "on_missing": on_missing, "n_requested": len(order_items),
            "resolution": res, "replayed": replayed, "not_replayed": skipped}


def write_resolution(r: dict, out_dir) -> Path:
    p = Path(out_dir) / "resolution.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2))
    return p
