import math

import numpy as np
import pytest

from conftest import frame, make_boxes, stream
from edge_signals import registry
from edge_signals.base import PairRecord, ReferenceBundle
from edge_signals.builtin.feature_drift import FeatureDrift, ledoit_wolf


def sig(name, **kw):
    return registry.get(name)(**kw)


def test_registry_declares_the_interface():
    for name, cls in registry.available().items():
        assert cls.family in ("quality", "drift", "context") and cls.granularity in ("frame", "pair", "window")
        assert set(cls.requires) <= {"preds", "features", "image"} and isinstance(cls.label_free, bool)
    assert set(registry.available()) >= {"consistency", "confidence", "platform_motion", "feature_drift"}


def test_consistency_identical_frames_and_fields():
    s = sig("consistency")
    v = s.update(PairRecord(frame(0), frame(1)))
    assert v.valid and v.value == pytest.approx(1.0, abs=1e-9) and v.extras["status"] == "MATCHED"
    assert set(v.extras) == {"alarm", "status", "G", "G_mc", "dx", "dy", "K", "N_t", "N_t1"}
    assert v.signal == "consistency" and "@" in v.version and v.reference_id is None


def test_consistency_status_rules_and_invalid_is_not_a_number():
    s = sig("consistency")
    empty = np.zeros((0, 6))
    v = s.update(PairRecord(frame(0, boxes=empty), frame(1, boxes=empty)))
    assert not v.valid and v.value is None and v.invalid_reason == "EMPTY_PAIR"
    v = s.update(PairRecord(frame(0), frame(1, boxes=empty)))          # ONE_SIDED is judged: consistency 0, valid
    assert v.valid and v.value == 0.0 and v.extras["status"] == "ONE_SIDED"
    assert all(x is None or isinstance(x, (int, float, str)) for x in v.to_dict()["extras"].values())


def test_platform_motion_shift_and_shared_result():
    b = make_boxes()
    pair = PairRecord(frame(0, boxes=b), frame(1, boxes=make_boxes(shift=(0.02, -0.01))))
    c = sig("consistency").update(pair)
    m = sig("platform_motion").update(pair)                   # reuses pair.cache
    assert "consistency_result" in pair.cache and m.valid
    assert m.extras["abs_dx"] == pytest.approx(0.02) and m.extras["abs_dy"] == pytest.approx(0.01)
    assert m.value == pytest.approx(math.hypot(0.02, 0.01)) and c.extras["dx"] == pytest.approx(0.02)
    nm = sig("platform_motion").update(PairRecord(frame(0), frame(1, boxes=np.zeros((0, 6)))))
    assert not nm.valid and nm.invalid_reason == "no_matched_pairs"


def test_confidence():
    s = sig("confidence")
    assert s.update(frame(0, conf=0.6)).value == pytest.approx(0.6)
    v = s.update(frame(0, boxes=np.zeros((0, 6))))
    assert not v.valid and v.value is None and v.invalid_reason == "no_boxes"


def _bundle(n=400, c=16, k=4, seed=0, backbone="bb1"):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, c) * np.linspace(2, 0.3, c) + 1.0
    recs = [frame(i, clip=f"c{i // 100}", feat=x[i], backbone=backbone) for i in range(n)]
    return FeatureDrift.fit_reference(recs, {"pca_dims": k}), x


def test_feature_drift_reference_and_distance():
    b, x = _bundle()
    s = FeatureDrift(reference=b)
    d_in = np.array([s.update(frame(i, feat=x[i])).value for i in range(400)])
    far = s.update(frame(0, feat=x[0] + 30 * np.ones(16))).value
    assert 0.5 * math.sqrt(4) < np.median(d_in) < 2 * math.sqrt(4) and far > 5 * np.median(d_in)
    assert b.meta["pca_dims"] == 4 and b.meta["backbone_params_id"] == "bb1" and b.meta["backbone_state_id"] == "bb1-s" and "valid_for" in b.meta
    assert s.update(frame(0, feat=x[0])).reference_id == b.id


def test_feature_drift_invalid_reasons_never_a_number():
    b, x = _bundle()
    s = FeatureDrift(reference=b)
    assert s.update(frame(0, feat=None)).invalid_reason == "no_features"
    assert s.update(frame(0, feat=x[0], backbone="other")).invalid_reason == "backbone_mismatch"
    assert FeatureDrift().update(frame(0, feat=x[0])).invalid_reason == "no_reference"


def test_reference_bundle_roundtrip_hash_and_tamper(tmp_path):
    b, _ = _bundle()
    p = b.save(tmp_path)
    b2 = ReferenceBundle.load(p)
    assert b2.id == b.id and np.array_equal(b2.arrays["components"], b.arrays["components"])
    bad = tmp_path / "feature_drift.0000000000000000.npz"
    p.rename(bad)
    with pytest.raises(ValueError, match="corrupt"):
        ReferenceBundle.load(bad)


def test_ledoit_wolf_matches_the_definition():
    rng = np.random.RandomState(1)
    x = rng.randn(60, 5) @ rng.randn(5, 5)
    x = x - x.mean(0)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    num = sum(np.sum((np.outer(r, r) - s) ** 2) for r in x) / n ** 2
    den = np.sum((s - mu * np.eye(p)) ** 2)
    shrink = min(num, den) / den
    cov, sh = ledoit_wolf(x)
    assert sh == pytest.approx(shrink, rel=1e-9) and np.allclose(cov, (1 - shrink) * s + shrink * mu * np.eye(p))


def test_reference_level_uses_windows_within_clips():
    b, _ = _bundle()
    s = FeatureDrift(reference=b)
    lv = s.reference_level("median", 30)
    assert lv is not None and 0.5 < lv < 4


def test_bn_stats_difference_is_logged_and_strict_mode_invalidates():
    b, x = _bundle()
    f = frame(0, feat=x[0])
    f.meta["backbone_state_id"] = "another-state"
    v = FeatureDrift(reference=b).update(f)
    assert v.valid and v.extras["bn_stats_match"] is False           # same learnable backbone, other BN statistics
    v = FeatureDrift({"backbone_match": "state"}, reference=b).update(f)
    assert not v.valid and v.invalid_reason == "backbone_state_mismatch"
