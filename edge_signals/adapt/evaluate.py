"""evaluate: a model on labelled lists (offline mAP - LABEL-REQUIRED) and the label-free consistency proxy on the most recent window.

mAP uses predictions at conf 0.001 (NMS 0.7, max_det 300); recall by box size counts predictions with conf >= recall_conf. The proxy is the
consistency window statistic of the last full window of the list's last clip, from predictions filtered at the deployment confidence
(0.25; identical to a conf 0.25 predict because NMS never lets a lower-confidence box suppress a higher one).
Writes <run>/evaluate/<tag>.json.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..base import FrameRecord
from ..inference import resolve_device
from .annotate import _xyxy, read_labels
from .common import clip_of, image_path, jdump, read_stems, rp, sha256_file, stage_dir
from .metrics import ap_single_class, recall_by_size


def predict_frames(model, images_root, stems, ev: dict, device):
    """[(stem, pred_xywhn(n,4), pred_xyxy(n,4), conf(n))] at the evaluation confidence."""
    out = []
    imgs = [str(image_path(images_root, s)) for s in stems]
    for i in range(0, len(imgs), 64):
        for r in model.predict(imgs[i:i + 64], conf=ev["map_conf"], iou=ev["map_iou"], max_det=ev["max_det"], imgsz=ev["imgsz"], batch=16, device=device,
                               half=False, verbose=False, stream=True):
            out.append((Path(r.path).stem, r.boxes.xywhn.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    return out


def proxy_last_window(cfg, preds, fps):
    """Consistency window statistic of the last full window of the last clip (label-free)."""
    from ..config import validate_signals, validate_trigger
    from ..engine import SignalEngine
    ev = cfg["evaluate"]
    last = clip_of(preds[-1][0])
    frames = [p for p in preds if clip_of(p[0]) == last]
    sc = validate_signals({"defaults": {"window_transitions": ev["proxy_window_transitions"], "aggregator": "median", "comparator": "previous", "change": "difference",
                                        "min_valid_fraction": 0.5}, "signals": {"consistency": {"measure": True, "trigger": False}}}, fps)
    recs = []
    for i, (st, xywhn, _, conf) in enumerate(frames):
        k = conf >= ev["proxy_conf"]
        boxes = np.concatenate([np.zeros((int(k.sum()), 1)), xywhn[k], conf[k][:, None]], 1) if k.any() else np.zeros((0, 6))
        recs.append(FrameRecord(t=i / fps, frame_id=st, boxes=boxes, clip_id=last))
    eng = SignalEngine(sc, validate_trigger({"trigger": {"rules": {}}}, sc), fps=fps)
    wins = eng.run(recs)["windows"]
    if not wins:
        return {"consistency_window_median": None, "reason": f"the last clip has fewer than {ev['proxy_window_transitions'] + 1} frames", "clip": last}
    w = wins[-1]
    return {"consistency_window_median": w.statistic, "window_id": w.window_id, "n_valid": w.n_valid, "clip": last,
            "window_transitions": ev["proxy_window_transitions"], "predictions_conf": ev["proxy_conf"],
            "mean_of_window_medians_last_clip": float(np.mean([x.statistic for x in wins if x.statistic is not None])) if wins else None}


def evaluate(cfg: dict, run, weights, evals: dict, images_root, labels_dir, tag: str, fps: float = 6.0) -> dict:
    from ultralytics import YOLO
    ev = cfg["evaluate"]
    weights = Path(weights)
    model = YOLO(str(weights))
    out = stage_dir(run, "evaluate")
    lab = rp(labels_dir)
    res = {"tag": tag, "weights": str(weights), "weights_sha256": sha256_file(weights), "settings": ev, "splits": {}}
    W, H = cfg.get("image_size_px", [1920, 1080])
    t0 = time.perf_counter()
    for name, lst in evals.items():
        stems = read_stems(lst)
        preds = predict_frames(model, images_root, stems, ev, resolve_device(ev.get("device", "auto")))
        frames = [(_xyxy(read_labels(lab / f"{st}.txt"), W, H), pb, pf) for st, _, pb, pf in preds]
        m50, m5095 = ap_single_class(frames)
        clips = sorted({clip_of(st) for st, *_ in preds})
        per = {}
        for c in clips:
            fv = [f for (st, *_), f in zip(preds, frames) if clip_of(st) == c]
            a50, a5095 = ap_single_class(fv)
            per[c] = {"n_frames": len(fv), "mAP50": a50, "mAP50_95": a5095}
        v = np.array([e["mAP50"] for e in per.values() if e["mAP50"] is not None])
        res["splits"][name] = {
            "n_frames": len(stems), "n_videos": len(clips),
            "LABEL_REQUIRED": {"pooled_mAP50": m50, "pooled_mAP50_95": m5095, "per_video_mAP50_median": float(np.median(v)) if len(v) else None,
                               "per_video_mAP50_q1": float(np.percentile(v, 25)) if len(v) else None, "per_video_mAP50_q3": float(np.percentile(v, 75)) if len(v) else None,
                               "recall_by_size": recall_by_size(frames, ev["size_bins_px"], 0.5, ev["recall_conf"]), "per_video": per},
            "LABEL_FREE": {"proxy_consistency_most_recent_window": proxy_last_window(cfg, preds, fps)}}
    res["eval_s"] = round(time.perf_counter() - t0, 2)
    jdump(res, out / f"{tag}.json")
    return res
