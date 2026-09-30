"""Edge inference with backbone feature capture in the SAME forward pass.

A forward hook on the last backbone layer of the model collects that layer's output; it is globally average pooled to
one vector per frame. No second forward pass is made. The hook layer index and output shape are recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import FrameRecord


def resolve_device(device):
    """'auto' -> GPU 0 if CUDA is available, else 'cpu'; anything else is returned unchanged."""
    if device == "auto":
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def installed_fork_commit():
    """Commit id recorded by pip for a `pip install git+https://...@<commit>` of ultralytics, or None."""
    try:
        from importlib.metadata import distribution
        raw = distribution("ultralytics").read_text("direct_url.json")
        return json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None
    except Exception:
        return None


def env_guard(env_cfg: dict) -> dict:
    """Abort unless ultralytics is the pinned fork (must run before any model load).
    $YOLOV12_DIR set: ultralytics must be imported from inside it. Unset: the pip-installed ultralytics must be the pinned commit."""
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SystemExit("[env_guard] PYTHONNOUSERSITE must be 1 (run through run.sh)")
    import ultralytics
    f = Path(ultralytics.__file__).resolve()
    if ultralytics.__version__ != env_cfg["ultralytics_version"]:
        raise SystemExit(f"[env_guard] ultralytics {ultralytics.__version__} != pinned {env_cfg['ultralytics_version']}")
    fork = os.environ.get(env_cfg.get("fork_dir_envvar", "YOLOV12_DIR"))
    info = {"ultralytics": ultralytics.__version__, "ultralytics_file": str(f)}
    if fork:
        if Path(fork).resolve() not in f.parents:
            raise SystemExit(f"[env_guard] ultralytics is imported from {f}, not from ${env_cfg.get('fork_dir_envvar')}={fork}")
        info["source"] = "fork_dir"
    else:
        commit = installed_fork_commit()
        if commit != env_cfg["fork_commit"]:
            raise SystemExit(f"[env_guard] installed ultralytics is not the pinned fork commit (found {commit}, pinned {env_cfg['fork_commit']}); "
                             f"install with: pip install -r requirements.txt")
        info.update(source="pip", fork_commit=commit)
    return info


def backbone_index(det_model) -> int:
    """Index of the last backbone layer of a YOLO DetectionModel (from its architecture dict)."""
    return len(det_model.yaml["backbone"]) - 1


def backbone_hash(det_model, last_index: int, buffers: bool) -> str:
    """sha256 over layers 0..last_index (sorted by name). buffers=False: learnable parameters only (identical across
    versions when the backbone is frozen); buffers=True: also the BatchNorm running statistics (which fine-tuning updates)."""
    import torch
    h = hashlib.sha256()
    sd = det_model.state_dict()
    pref = tuple(f"model.{i}." for i in range(last_index + 1))
    for k in sorted(sd):
        if k.startswith(pref) and (buffers or not k.endswith(("running_mean", "running_var", "num_batches_tracked"))):
            t = sd[k].detach().cpu().contiguous()
            h.update(k.encode())
            h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


class FeatureHook:
    """Forward hook that keeps the global average pool of a module's output for every forward call, in call order."""

    def __init__(self, module):
        self.vectors: List[np.ndarray] = []
        self.shape: Optional[tuple] = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        out = output[0] if isinstance(output, (list, tuple)) else output
        if self.shape is None:
            self.shape = tuple(out.shape[1:])            # per frame (channels, height, width)
        self.vectors.append(out.float().mean(dim=(2, 3)).detach().cpu().numpy())

    def pop(self) -> np.ndarray:
        v = np.concatenate(self.vectors, 0) if self.vectors else np.zeros((0, 0))
        self.vectors = []
        return v

    def remove(self):
        self.handle.remove()


class Detector:
    def __init__(self, weights, predict_cfg: dict, env_cfg: dict, want_features: bool = True):
        self.env = env_guard(env_cfg)
        from ultralytics import YOLO
        self.p = dict(predict_cfg)
        self.p["device"] = resolve_device(self.p["device"])
        self.model = YOLO(str(weights))
        dm = self.model.model
        self.hook_index = backbone_index(dm)
        self.backbone_params_id = backbone_hash(dm, self.hook_index, buffers=False)
        self.backbone_state_id = backbone_hash(dm, self.hook_index, buffers=True)
        self.hook = FeatureHook(dm.model[self.hook_index]) if want_features else None
        self.weights_sha256 = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        # ultralytics runs one extra warm-up forward on the first predict call; take it here so it never enters a stream
        self.model.predict(np.zeros((360, 640, 3), dtype=np.uint8), imgsz=self.p["imgsz"], device=self.p["device"],
                           half=self.p["half"], verbose=False)
        if self.hook:
            self.hook.pop()
            self.hook.shape = None            # the first real forward defines the recorded shape

    @property
    def hook_shape(self):
        return self.hook.shape if self.hook else None

    def infer(self, images: Sequence, ids: Sequence[str], t0: float = 0.0, fps: float = 6.0, clip_ids=None,
              device_id: str = "device", model_version: str = "model") -> List[FrameRecord]:
        """One predict call over `images` (paths or arrays); returns FrameRecords with boxes and pooled features."""
        p = self.p
        if self.hook:
            self.hook.pop()
        tic = time.perf_counter()
        res = self.model.predict(list(images), conf=p["conf"], iou=p["iou"], max_det=p["max_det"], imgsz=p["imgsz"],
                                 batch=p["batch"], device=p["device"], half=p["half"], agnostic_nms=p["agnostic_nms"],
                                 verbose=False)
        self.last_infer_s = time.perf_counter() - tic
        feats = self.hook.pop() if self.hook else None
        if feats is not None and len(feats) != len(res):
            raise RuntimeError(f"hook captured {len(feats)} vectors for {len(res)} frames")
        out = []
        for i, r in enumerate(res):
            b = r.boxes
            if len(b):
                arr = np.concatenate([b.cls.cpu().numpy()[:, None], b.xywhn.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
            else:
                arr = np.zeros((0, 6))
            out.append(FrameRecord(t=t0 + i / fps, frame_id=str(ids[i]), boxes=arr, clip_id=None if clip_ids is None else clip_ids[i],
                                   device_id=device_id, model_version=model_version,
                                   features=None if feats is None else feats[i].astype(np.float64),
                                   meta={"backbone_params_id": self.backbone_params_id, "backbone_state_id": self.backbone_state_id}))
        return out
