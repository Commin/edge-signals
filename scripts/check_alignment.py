"""Measure how the decoded frames of the ORIGINAL videos line up with the labelled images (stride and offset), per video.

    ./run.sh check-alignment --videos-dir DIR --frames-dir /data/frames [--stream NAME | --split NAME] [--codes CODE ...] [--out alignment.json]

The videos are found by the replay order file (same rules as `stream --videos-dir`). For each video the labelled image k (stem `<video>-<k>`) is
compared with decoded frame  offset + stride * k  for every stride 1..10 and offset 0..stride-1 (both sides converted to grey and resized to
320x180; mean PSNR and mean absolute difference over all labelled images of the video). Reported per video: the best (stride, offset), its PSNR and
mean absolute difference, the runner-up and the margin over it (dB), and labelled images x stride against the video's frame count.
Exit code 1 if any video's best candidate is not (stride 5, offset 0).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZE = (320, 180)
STRIDES = range(1, 11)


def small_gray(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA)


def decode_all(path):
    """All frames of the video, grey and reduced (uint8, N x 180 x 320), and the true decoded frame count."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(small_gray(fr))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return np.stack(out), fps


def score_candidates(labelled: np.ndarray, ks, dec: np.ndarray, strides=STRIDES, min_covered=0.8):
    """{(stride, offset): (mean PSNR, mean abs diff, n_compared)} for every hypothesis whose frames exist for >= min_covered of the labelled images."""
    lab = labelled.astype(np.float32)
    res = {}
    for s in strides:
        for o in range(s):
            idx = o + s * np.asarray(ks)
            ok = idx < len(dec)
            if ok.sum() < min_covered * len(ks) or ok.sum() == 0:
                continue
            d = dec[idx[ok]].astype(np.float32) - lab[ok]
            mse = (d * d).mean(axis=(1, 2))
            psnr = np.where(mse == 0, 100.0, 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12)))
            res[(s, o)] = (float(psnr.mean()), float(np.abs(d).mean()), int(ok.sum()))
    return res


def check_video(video_path, frames_dir, stems):
    import cv2
    ks, labs = [], []
    for st in stems:
        m = re.search(r"-(\d+)$", st)
        im = cv2.imread(str(Path(frames_dir) / f"{st}.jpg"))
        if m is None or im is None:
            continue
        ks.append(int(m.group(1)))
        labs.append(small_gray(im))
    dec, fps = decode_all(video_path)
    sc = score_candidates(np.stack(labs), ks, dec)
    rank = sorted(sc, key=lambda c: -sc[c][0])
    best, second = rank[0], rank[1] if len(rank) > 1 else None
    return {"video": Path(video_path).stem, "file": str(video_path), "n_labelled": len(ks), "n_frames_decoded": int(len(dec)), "fps": round(float(fps), 3),
            "best_stride": best[0], "best_offset": best[1], "psnr_db": round(sc[best][0], 2), "mad": round(sc[best][1], 2),
            "second": None if second is None else {"stride": second[0], "offset": second[1], "psnr_db": round(sc[second][0], 2), "mad": round(sc[second][1], 2)},
            "margin_db": None if second is None else round(sc[best][0] - sc[second][0], 2),
            "at_5_0": None if (5, 0) not in sc else {"psnr_db": round(sc[(5, 0)][0], 2), "mad": round(sc[(5, 0)][1], 2)},
            "at_5_pm1": {str(o): round(sc[(5, o)][0], 2) for o in (1, 4) if (5, o) in sc},
            "labelled_x_best_stride": len(ks) * best[0], "labelled_x_5": len(ks) * 5}


def _one(job):
    code, f, frames_dir, stems, split = job
    row = check_video(f, frames_dir, stems)
    row["code"], row["split"] = code, split
    return row


def stems_by_video(frames_dir=None, mapping=None):
    import csv
    by = {}
    for r in csv.DictReader(open(mapping)):
        by.setdefault(r["video"], []).append(r["image_stem"])
    return by


def main(argv=None):
    from edge_signals import replay
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--order", default=str(ROOT / "configs" / "replay_order.yaml"))
    ap.add_argument("--mapping", default=str(ROOT / "data" / "mapping" / "frame_map.csv"), help="video -> labelled image stems")
    ap.add_argument("--stream")
    ap.add_argument("--split")
    ap.add_argument("--codes", nargs="*", help="only these video codes")
    ap.add_argument("--out", help="write the per-video table as JSON")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    order = replay.load_order(a.order)
    items = replay.ordered_codes(order, a.stream, a.split) if (a.stream or a.split) else [{"code": c, "split": s} for s, cs in order["splits"].items() for c in cs]
    if a.codes:
        items = [i for i in items if i["code"] in a.codes]
    r = replay.resolve(items, a.videos_dir, "skip")
    by = stems_by_video(mapping=a.mapping)
    jobs = [(x["code"], x["file"]) for x in r["replayed"]]
    for m in r["not_replayed"]:
        print(f"[warning] {m['code']}: {m['reason']} - not checked", file=sys.stderr)

    jobs = [(c, f, a.frames_dir, by[c], r["resolution"][c]["split"]) for c, f in jobs]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as p:
            rows = p.map(_one, jobs)
    else:
        rows = [_one(j) for j in jobs]
    print(f"{'video':22s} {'split':14s} {'n_lab':>5s} {'frames':>6s} {'stride':>6s} {'offset':>6s} {'PSNR':>6s} {'MAD':>5s} {'margin dB':>9s}  runner-up")
    for x in rows:
        sec = x["second"]
        print(f"{x['video']:22s} {x['split']:14s} {x['n_labelled']:5d} {x['n_frames_decoded']:6d} {x['best_stride']:6d} {x['best_offset']:6d} {x['psnr_db']:6.2f} {x['mad']:5.2f} "
              f"{x['margin_db']:9.2f}  ({sec['stride']},{sec['offset']}) {sec['psnr_db']:.2f} dB")
    bad = [(x["video"], x["best_stride"], x["best_offset"]) for x in rows if (x["best_stride"], x["best_offset"]) != (5, 0)]
    print("\nvideos whose best candidate is not (stride 5, offset 0):", bad if bad else "none")
    if a.out:
        Path(a.out).write_text(json.dumps({"videos": rows, "not_checked": r["not_replayed"], "not_default": bad}, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
