"""Write a few synthetic frames (sea gradient, a moving 'ship', small camera shake) named <clip>-<7-digit index>.jpg.

    ./run.sh synthetic-frames --out DIR [--clips 2] [--frames 40] [--seed 0]

For smoke tests of the pipeline only: the images carry no real content, the detector will find little or nothing in them.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make(out: Path, clips: int = 2, frames: int = 40, seed: int = 0, size=(640, 384)):
    rng = np.random.RandomState(seed)
    w, h = size
    out.mkdir(parents=True, exist_ok=True)
    yy = np.linspace(0, 1, h)[:, None, None]
    written = []
    for c in range(clips):
        x0, y0 = rng.randint(50, w // 2), rng.randint(h // 3, h // 2)
        for i in range(frames):
            img = np.zeros((h, w, 3), dtype=np.float32)
            img[..., 0] = 40 + 60 * yy[..., 0]
            img[..., 1] = 90 + 60 * yy[..., 0]
            img[..., 2] = 140 + 40 * yy[..., 0]
            dx, dy = rng.randint(-2, 3), rng.randint(-2, 3)                       # camera shake
            x, y = int(x0 + 4 * i + dx), int(y0 + dy)
            img[y:y + 22, x:x + 70] = (200, 200, 205)                             # hull
            img[y - 12:y, x + 20:x + 45] = (230, 230, 235)                        # superstructure
            img += rng.normal(0, 4, img.shape)
            p = out / f"synth{c}-{i:07d}.jpg"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(p, quality=90)
            written.append(p)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=2)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    n = len(make(Path(a.out), a.clips, a.frames, a.seed))
    print(f"{n} synthetic frames written to {a.out}")


if __name__ == "__main__":
    main()
