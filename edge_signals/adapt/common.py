"""Shared helpers of the adaptation stages: config with --set overrides, frame lists, JSON, hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = (".jpg", ".jpeg", ".png")


def rp(p) -> Path:
    """Resolve a config path: absolute stays, relative is taken from the package root."""
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else ROOT / p


def load_cfg(path="configs/adaptation.yaml", sets: Optional[List[str]] = None) -> dict:
    cfg = yaml.safe_load(open(rp(path))) or {}
    for s in sets or []:
        key, _, val = s.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got '{s}'")
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def read_stems(list_path) -> List[str]:
    """Frame names of a list file (one per line; a path or a bare stem); order kept."""
    out = []
    for line in open(rp(list_path)):
        s = line.strip()
        if s:
            out.append(Path(s).stem if Path(s).suffix.lower() in IMG_EXT else s)
    return out


def image_path(root, stem) -> Path:
    return Path(root) / f"{stem}.jpg"


def clip_of(stem: str) -> str:
    return stem.rsplit("-", 1)[0]


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def jload(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def stage_dir(run, name) -> Path:
    d = Path(run) / name
    d.mkdir(parents=True, exist_ok=True)
    return d
