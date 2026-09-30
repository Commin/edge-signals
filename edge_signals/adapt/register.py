"""register: add a model version to registry.json (version, parent, request id, recipe, SHA256, time, reference ids, bn_stats_match)."""
from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .common import jdump, jload, rp, sha256_file, sha256_obj, stage_dir


def next_version(registry: dict, parent_version: Optional[str], family: Optional[str] = None) -> str:
    fam = family or (re.sub(r"\.v\d+$", "", parent_version) if parent_version else "model")
    ks = [int(m.group(1)) for v in registry["versions"] if (m := re.match(rf"^{re.escape(fam)}\.v(\d+)$", v["version"]))]
    return f"{fam}.v{(max(ks) + 1) if ks else 1}"


def build_entry(version, parent, request_id, recipe, weights_sha256, weights_path, backbone_ids, reference, stream_t=None, retrain_summary=None,
                evaluation=None) -> dict:
    """reference: {"feature_drift": {"reference_id", "backbone_params_id", "backbone_state_id"}} or None."""
    ref = (reference or {}).get("feature_drift") or {}
    entry = {"version": version, "parent": parent, "request_id": request_id, "recipe": recipe, "recipe_sha256": sha256_obj(recipe),
             "weights": str(weights_path), "weights_sha256": weights_sha256, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "stream_t": stream_t, "backbone_params_id": backbone_ids["params"], "backbone_state_id": backbone_ids["state"],
             "reference_ids": {"feature_drift": ref.get("reference_id")},
             "backbone_params_match": (backbone_ids["params"] == ref["backbone_params_id"]) if ref else None,
             "bn_stats_match": (backbone_ids["state"] == ref["backbone_state_id"]) if ref else None}
    if retrain_summary:
        entry["retrain"] = retrain_summary
    if evaluation:
        entry["evaluation"] = evaluation
    return entry


def register(cfg: dict, run, weights, parent=None, request_id=None, recipe_json=None, reference_path=None, registry_path=None,
             stream_t=None, family=None, device="cpu") -> dict:
    """Copy `weights` (optimizer stripped) to <run>/models/<version>.pt and append the entry."""
    from ..base import ReferenceBundle
    from ..inference import backbone_hash, backbone_index
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import strip_optimizer
    run = Path(run)
    reg_path = Path(registry_path) if registry_path else run / cfg.get("registry", {}).get("path", "registry.json")
    reg = jload(reg_path, {"versions": []})
    version = next_version(reg, parent, family)
    models = stage_dir(run, "models")
    dst = models / f"{version}.pt"
    shutil.copy2(weights, dst)
    strip_optimizer(str(dst))
    m = YOLO(str(dst)).model
    idx = backbone_index(m)
    ids = {"params": backbone_hash(m, idx, buffers=False), "state": backbone_hash(m, idx, buffers=True)}
    reference = None
    if reference_path:
        b = ReferenceBundle.load(rp(reference_path))
        reference = {"feature_drift": {"reference_id": b.id, "backbone_params_id": b.meta["backbone_params_id"],
                                       "backbone_state_id": b.meta["backbone_state_id"]}}
    rj = jload(recipe_json, {}) if recipe_json else {}
    entry = build_entry(version, parent, request_id, rj.get("recipe", {}), sha256_file(dst), dst, ids, reference, stream_t,
                        {k: rj.get(k) for k in ("train_s", "per_phase_s", "peak_gpu_MiB", "seed")} if rj else None)
    reg["versions"].append(entry)
    jdump(reg, reg_path)
    return entry
