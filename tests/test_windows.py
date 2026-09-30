import numpy as np
import pytest

from edge_signals.base import Signal, SignalValue
from edge_signals.windows import WindowTracker, aggregate


class Dummy(Signal):
    name, family, granularity, higher_is_healthier = "dummy", "quality", "frame", True


def sv(v, t=0.0, valid=True):
    return SignalValue("dummy", "1", v if valid else None, valid, None if valid else "x", t, "d", "m", None)


def tracker(**kw):
    cfg = dict(window_transitions=3, aggregator="mean", comparator="previous", rolling_windows=2, change="difference")
    cfg.update(kw)
    h = cfg.pop("healthy", True)
    d = Dummy()
    d.higher_is_healthier = h
    return WindowTracker(d, cfg)


def feed(tr, vals, clip="a"):
    out = []
    for v in vals:
        w = tr.push(sv(v), clip)
        if w:
            out.append(w)
    return out


def test_aggregators():
    v = list(range(1, 21))                      # 20 values: worst 10 % = 2 values
    assert aggregate(v, "mean", True) == 10.5 and aggregate(v, "median", True) == 10.5
    assert aggregate(v, "worst10", True) == 1.5                # lowest two when higher is healthier
    assert aggregate(v, "worst10", False) == 19.5              # highest two when higher is worse
    assert aggregate([5.0], "worst10", True) == 5.0            # at least one value


def test_previous_and_difference_sign():
    ws = feed(tracker(), [1, 1, 1, 0.6, 0.6, 0.6])
    assert ws[0].change is None and ws[0].reference is None
    assert ws[1].reference == 1 and ws[1].change == pytest.approx(0.4)      # a drop is a positive change
    ws = feed(tracker(healthy=False), [1, 1, 1, 1.5, 1.5, 1.5])
    assert ws[1].change == pytest.approx(0.5)                               # a rise is degradation for a higher-is-worse signal


def test_ratio_change():
    ws = feed(tracker(change="ratio"), [1, 1, 1, 0.5, 0.5, 0.5])
    assert ws[1].change == pytest.approx(0.5)


def test_anchor_stays_and_reset_restarts():
    tr = tracker(comparator="anchor")
    ws = feed(tr, [1, 1, 1, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5])
    assert [w.reference for w in ws] == [None, 1, 1] and ws[2].change == pytest.approx(0.5)
    tr.reset_reference()
    ws = feed(tr, [0.4, 0.4, 0.4, 0.2, 0.2, 0.2])
    assert ws[0].reference is None and ws[1].reference == pytest.approx(0.4)


def test_rolling_needs_M_windows():
    ws = feed(tracker(comparator="rolling", rolling_windows=2), [1, 1, 1, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5])
    assert ws[1].reference is None and ws[1].invalid_reason.startswith("no_change")
    assert ws[2].reference == pytest.approx(0.9) and ws[2].change == pytest.approx(0.4)


def test_invalid_values_are_never_filled():
    tr = tracker(min_valid_fraction=0.5)
    ws = []
    for v, ok in [(0.9, True), (None, False), (0.7, True)]:
        w = tr.push(sv(v, valid=ok), "a")
        ws = [w] if w else ws
    w = ws[0]
    assert w.n_items == 3 and w.n_valid == 2 and w.statistic == pytest.approx(0.8)     # mean of the valid ones only
    tr2 = tracker(min_valid_fraction=0.9)
    for v, ok in [(0.9, True), (None, False), (0.7, True)]:
        last = tr2.push(sv(v, valid=ok), "a")
    assert last.valid is False and last.statistic is None and last.invalid_reason == "too_few_valid_values"


def test_windows_do_not_cross_clips_but_comparison_state_does():
    tr = tracker()
    first = feed(tr, [1, 1, 1, 1, 1], clip="a")          # 5 values: one full window + 2 left over
    tr.close_clip()
    assert tr.dropped_partial == 1 and len(first) == 1
    second = feed(tr, [0.5, 0.5, 0.5], clip="b")           # first window of the next clip is compared with the previous clip's
    assert second[0].reference == 1 and second[0].change == pytest.approx(0.5)


def test_healthy_change_distribution_and_delta():
    import numpy as np
    from conftest import stream
    from edge_signals.healthy import delta_from, derive_changes
    clips = {f"c{i}": stream(16, f"c{i}") for i in range(3)}
    cfg = {"defaults": {"window_transitions": 5, "aggregator": "median", "comparator": "previous", "rolling_windows": 2, "change": "difference"},
           "signals": {"consistency": {"measure": True, "trigger": False}}}
    b = derive_changes(clips, cfg, 6.0)
    assert len(b.arrays["previous"]) == 6 and len(b.arrays["anchor"]) == 6              # 3 windows per clip: 2 changes each
    assert b.meta["window_transitions"] == 5 and b.meta["aggregator"] == "median" and delta_from(b, "previous") == pytest.approx(0.0, abs=1e-9)
