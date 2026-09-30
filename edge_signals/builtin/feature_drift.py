"""feature_drift (drift, frame): input drift of the edge model's backbone features.

Per frame, the globally pooled vector of the last backbone layer (captured in the same inference call, no second
forward pass) is projected on a PCA fitted on a reference set (the model's training frames) and scored by the
Mahalanobis distance under a Ledoit-Wolf shrunk covariance. Higher distance = further from the training data.

The reference bundle stores the backbone identity twice: backbone_params_id (sha256 of the learnable backbone weights,
identical across model versions because the backbone is frozen in adaptation) and backbone_state_id (also the BatchNorm
running statistics, which fine-tuning does update). The bundle applies to every version with the same backbone_params_id;
a different one gets INVALID values (backbone_mismatch), never a number. params.backbone_match: state makes the check strict.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import FrameRecord, ReferenceBundle, Signal
from ..registry import register


def ledoit_wolf(x: np.ndarray):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards mu*I. x: (n, p), already centred. -> (cov, shrinkage)."""
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2)
    # exact form: mean_k ||x_k x_k^T||_F^2 - ||S||_F^2 , divided by n
    b2 = (np.sum(np.sum(x ** 2, axis=1) ** 2) / n - np.sum(s ** 2)) / n
    b2 = min(b2, d2)
    shrink = 0.0 if d2 == 0 else b2 / d2
    return (1 - shrink) * s + shrink * mu * np.eye(p), float(shrink)


def window_stat(values: np.ndarray, aggregator: str, higher_is_healthier: bool) -> float:
    from ..windows import aggregate
    return aggregate(values, aggregator, higher_is_healthier)


@register
class FeatureDrift(Signal):
    name = "feature_drift"
    version = "1"
    family = "drift"
    granularity = "frame"
    requires = frozenset({"features"})
    higher_is_healthier = False
    label_free = True

    def __init__(self, params=None, reference: Optional[ReferenceBundle] = None):
        super().__init__(params, reference)
        if reference is not None:
            a = reference.arrays
            self._mean, self._comp, self._prec = a["mean"], a["components"], a["precision"]

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit_reference(cls, records, params=None) -> ReferenceBundle:
        """records: FrameRecords with .features and .meta['backbone_id'] (training frames of the model)."""
        params = params or {}
        recs = [r for r in records if r.features is not None]
        if not recs:
            raise ValueError("feature_drift.fit_reference: no record carries features")
        ids = {(r.meta.get("backbone_params_id"), r.meta.get("backbone_state_id")) for r in recs}
        if len(ids) != 1 or None in next(iter(ids)):
            raise ValueError(f"records must come from one model / backbone (found {ids})")
        params_id, state_id = next(iter(ids))
        x = np.stack([np.asarray(r.features, dtype=np.float64) for r in recs])
        k = int(params.get("pca_dims", 32))
        mean = x.mean(0)
        u, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        k = min(k, len(s))
        comp = vt[:k]
        z = (x - mean) @ comp.T
        cov, shrink = ledoit_wolf(z)
        prec = np.linalg.inv(cov)
        d = np.sqrt(np.einsum("ni,ij,nj->n", z, prec, z))
        clips = sorted({r.clip_id or "" for r in recs})
        clip_idx = np.array([clips.index(r.clip_id or "") for r in recs], dtype=np.int32)
        meta = {"signal": "feature_drift", "pca_dims": k, "feature_dim": int(x.shape[1]), "n_frames": int(len(recs)),
                "n_clips": len(clips), "shrinkage": shrink, "backbone_params_id": params_id, "backbone_state_id": state_id,
                "hook_layer": params.get("hook_layer"), "hook_output_shape": params.get("hook_output_shape"),
                "pooling": "global average over spatial positions",
                "explained_variance_ratio": float((s[:k] ** 2).sum() / (s ** 2).sum()),
                "reference_distance_median": float(np.median(d)), "reference_distance_p95": float(np.percentile(d, 95)),
                "valid_for": "every model version whose learnable backbone weights are identical (backbone_params_id; frozen in adaptation). "
                                 "BatchNorm running statistics are NOT frozen by fine-tuning: a version with a different backbone_state_id "
                                 "shares the parameters but its features are shifted slightly (logged as bn_stats_match = false)."}
        return ReferenceBundle("feature_drift", {"mean": mean, "components": comp, "precision": prec,
                                                 "reference_distances": d, "reference_clip_index": clip_idx}, meta)

    # ------------------------------------------------------------------ scoring
    def distance(self, vec: np.ndarray) -> float:
        z = self._comp @ (np.asarray(vec, dtype=np.float64) - self._mean)
        return float(np.sqrt(z @ self._prec @ z))

    def update(self, record: FrameRecord):
        if self.reference is None:
            return self.invalid(record, "no_reference")
        if record.features is None:
            return self.invalid(record, "no_features")
        m = self.reference.meta
        if record.meta.get("backbone_params_id") != m["backbone_params_id"]:
            return self.invalid(record, "backbone_mismatch")          # different learnable backbone: the reference does not apply
        same_state = record.meta.get("backbone_state_id") == m["backbone_state_id"]
        if self.params.get("backbone_match", "params") == "state" and not same_state:
            return self.invalid(record, "backbone_state_mismatch")
        return self.value(record, self.distance(record.features), {"bn_stats_match": same_state})

    def reference_level(self, aggregator: str, window: int) -> Optional[float]:
        """Median over the reference windows (W consecutive reference frames, never across clips) of the window statistic."""
        if self.reference is None:
            return None
        d, ci = self.reference.arrays["reference_distances"], self.reference.arrays["reference_clip_index"]
        stats = []
        for c in np.unique(ci):
            v = d[ci == c]
            for i in range(len(v) // window):
                stats.append(window_stat(v[i * window:(i + 1) * window], aggregator, self.higher_is_healthier))
        return float(np.median(stats)) if stats else None
