"""annotate: labels for the collected frames.

mode "provided": existing labels (simulated annotation), with an optional injected delay (logged, added to the logical clock; nothing sleeps).
mode "pseudo"  : predictions of a teacher model (weights path configurable, mounted, not shipped) at a confidence threshold.
Replay frames always take their existing labels. Writes <run>/annotate/{labels/*.txt, annotate.json}. Pseudo-label precision / recall
against ground truth is computed only if a ground-truth directory is given (LABEL-REQUIRED, reported separately).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .common import image_path, jdump, jload, rp, sha256_file, stage_dir


def _xyxy(g: np.ndarray, W: int, H: int) -> np.ndarray:
    if not len(g):
        return np.zeros((0, 4))
    return np.stack([(g[:, 1] - g[:, 3] / 2) * W, (g[:, 2] - g[:, 4] / 2) * H, (g[:, 1] + g[:, 3] / 2) * W, (g[:, 2] + g[:, 4] / 2) * H], 1)


def read_labels(path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 5))
    return np.array([[float(x) for x in l.split()[:5]] for l in open(p) if l.strip()]).reshape(-1, 5)


def match_counts(gt_xyxy, pred_xyxy, iou_thr: float) -> int:
    from .metrics import box_iou
    iou = box_iou(gt_xyxy, pred_xyxy)
    hit, used, n = np.zeros(len(gt_xyxy), bool), np.zeros(len(pred_xyxy), bool), 0
    for v, a, b in sorted(((iou[a, b], a, b) for a in range(len(gt_xyxy)) for b in range(len(pred_xyxy)) if iou[a, b] >= iou_thr), reverse=True):
        if not hit[a] and not used[b]:
            hit[a], used[b] = True, True
            n += 1
    return n


def annotate(cfg: dict, run, images_root, mode=None, labels_dir=None, replay_labels_dir=None, teacher=None, check_labels_dir=None) -> dict:
    acfg = cfg["annotate"]
    mode = mode or acfg["mode"]
    col = jload(Path(run) / "collect" / "collect.json")
    if col is None:
        raise SystemExit("run collect first (missing collect/collect.json)")
    if col.get("status") == "insufficient_data":
        raise SystemExit(f"collect logged 'insufficient data' ({col['n_adapt']} new labelled frames < minimum {col['min_new_frames']}): no annotation, no retraining")
    adapt = [l.strip() for l in open(Path(run) / "collect" / "adapt.txt") if l.strip()]
    replay = [l.strip() for l in open(Path(run) / "collect" / "replay.txt") if l.strip()]
    out = stage_dir(run, "annotate")
    lab = out / "labels"
    if lab.exists():
        shutil.rmtree(lab)
    lab.mkdir(parents=True)
    t0 = time.perf_counter()
    info = {"mode": mode, "n_adapt": len(adapt), "n_replay": len(replay)}
    prov = rp(labels_dir or acfg["provided"]["labels_dir"])
    rep_dir = rp(replay_labels_dir or acfg.get("replay_labels_dir") or prov)
    missing = [s for s in replay if not (rep_dir / f"{s}.txt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} replay frames have no label file in {rep_dir} (first: {missing[0]})")
    for s in replay:
        shutil.copy2(rep_dir / f"{s}.txt", lab / f"{s}.txt")
    if mode == "provided":
        missing = [s for s in adapt if not (prov / f"{s}.txt").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} adaptation frames have no provided label file in {prov} (first: {missing[0]})")
        for s in adapt:
            shutil.copy2(prov / f"{s}.txt", lab / f"{s}.txt")
        d = acfg["provided"]
        delay = float(d.get("delay_s_fixed", 0.0)) + float(d.get("delay_s_per_frame", 0.0)) * len(adapt)
        info.update(labels_dir=str(prov), simulated_annotation_delay_s=delay,
                    note="simulated annotation: the delay is logged and added to the logical clock, nothing sleeps")
    elif mode == "pseudo":
        from ultralytics import YOLO
        from ..inference import resolve_device
        p = acfg["pseudo"]
        tw = rp(teacher or p["teacher_weights"])
        if not tw.exists():
            raise SystemExit(f"teacher weights not found: {tw} (mount them, e.g. at /models; they are not shipped)")
        model = YOLO(str(tw))
        imgs = [str(image_path(images_root, s)) for s in adapt]
        n_box = 0
        stats = {"tp": 0, "n_pred": 0, "n_gt": 0}
        chk = rp(check_labels_dir) if check_labels_dir else None
        W, H = cfg.get("image_size_px", [1920, 1080])
        for i in range(0, len(imgs), 64):
            for r in model.predict(imgs[i:i + 64], conf=p["conf"], iou=p["iou"], imgsz=p["imgsz"], device=resolve_device(p.get("device", "auto")), batch=16,
                                   half=False, verbose=False, stream=True):
                st = Path(r.path).stem
                xywhn = r.boxes.xywhn.cpu().numpy()
                (lab / f"{st}.txt").write_text("".join(f"0 {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for a, b, w, h in xywhn))
                n_box += len(xywhn)
                if chk is not None:
                    g = read_labels(chk / f"{st}.txt")
                    gb, pb = _xyxy(g, W, H), r.boxes.xyxy.cpu().numpy()
                    stats["tp"] += match_counts(gb, pb, 0.5)
                    stats["n_pred"] += len(pb)
                    stats["n_gt"] += len(gb)
        info.update(teacher={"weights_sha256": sha256_file(tw), "conf": p["conf"], "iou": p["iou"]}, n_pseudo_boxes=n_box)
        if chk is not None:
            info["pseudo_label_quality_LABEL_REQUIRED"] = {"iou": 0.5, "precision": stats["tp"] / max(1, stats["n_pred"]),
                                                          "recall": stats["tp"] / max(1, stats["n_gt"]), "n_pred": stats["n_pred"], "n_gt": stats["n_gt"]}
    else:
        raise SystemExit(f"annotate.mode must be provided | pseudo (got {mode})")
    info["annotate_s"] = round(time.perf_counter() - t0, 2)
    jdump(info, out / "annotate.json")
    return info
