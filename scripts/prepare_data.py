"""Lay out raw frames as <target>/<clip>-<index:07d>.jpg (for someone who has the original frames, not the released data assets).

    ./run.sh prepare-data --config configs/data.yaml [--source-dir DIR] [--url URL] [--sha256 HEX]

The released assets (weights, labelled frames, videos) are downloaded with `./run.sh fetch-assets` instead. The dataset must be obtained under its own licence.
"""
import argparse
import hashlib
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG = (".jpg", ".jpeg", ".png")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, expect, workdir: Path) -> Path:
    dst = workdir / Path(url.split("?")[0]).name
    urllib.request.urlretrieve(url, dst)
    if expect and sha256(dst) != expect.lower():
        raise SystemExit(f"sha256 mismatch for {dst.name}: got {sha256(dst)}, expected {expect}")
    return dst


def extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):                # tar first: a tar ending in a zip member also looks like a zip
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if Path(m.name).is_absolute() or ".." in Path(m.name).parts:
                    raise SystemExit(f"unsafe path in archive: {m.name}")
            t.extractall(into)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():                       # refuse path traversal
                if Path(n).is_absolute() or ".." in Path(n).parts:
                    raise SystemExit(f"unsafe path in archive: {n}")
            z.extractall(into)
    else:
        raise SystemExit(f"{archive.name}: not a zip / tar archive")
    return into


def lay_out(src: Path, target: Path, regex: str, reindex: bool) -> int:
    rx = re.compile(regex)
    clips = {}
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMG and not p.name.startswith("."):
            m = rx.match(p.stem)
            if m:
                clips.setdefault(m.group("clip"), []).append((int(m.group("n")), p))
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip, items in clips.items():
        for i, (num, p) in enumerate(sorted(items)):
            shutil.copy2(p, target / f"{clip}-{(i if reindex else num):07d}.jpg")
            n += 1
    return n


def check_lists(target: Path, lists):
    missing = []
    for lst in lists:
        for line in open(ROOT / lst if not Path(lst).is_absolute() else lst):
            s = line.strip()
            if s and not (target / f"{s}.jpg").exists():
                missing.append(s)
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--source-dir")
    ap.add_argument("--url")
    ap.add_argument("--sha256")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    src = a.source_dir or cfg["source"].get("local_dir")
    url = a.url or cfg["source"].get("url")
    target = Path(cfg["target_dir"])
    target = target if target.is_absolute() else ROOT / target
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if not src:
            if not url:
                sys.exit("nothing to prepare: set source.url or source.local_dir in configs/data.yaml (or pass --url / --source-dir)")
            src = extract(fetch(url, a.sha256 or cfg["source"].get("sha256"), tmp), tmp / "x")
        n = lay_out(Path(src), target, cfg["input_name_regex"], cfg["reindex"])
    missing = check_lists(target, cfg.get("frame_lists", []))
    print(f"{n} frames written to {target}; {len(missing)} frames named in the frame lists are missing")
    if missing:
        print("first missing:", missing[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
