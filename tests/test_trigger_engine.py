import copy
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import HOOKS, STEP_RULE, frame, make_boxes, sig_cfg, stream
from edge_signals.base import ReferenceBundle
from edge_signals.config import validate_signals, validate_trigger
from edge_signals.engine import SignalEngine

SUST_RULE = {"type": "relative_drop", "enabled": True, "stage": "confirmation", "comparator": "anchor", "delta": 0.2, "n_consecutive": 2,
             "cooldown_windows": 0}


def make(rules, tmp_path=None, healthy=None, **over):
    sc = validate_signals(sig_cfg(**over), fps=6.0)
    tc = validate_trigger({"trigger": {"rules": {**rules, **HOOKS}}}, sc)
    return SignalEngine(sc, tc, out_dir=tmp_path, fps=6.0, healthy=healthy)


def good(n, clip="clipA", start=0):
    return stream(n, clip, start)


def bad(n, clip="clipA", start=0):
    """Frames whose boxes jump: consistency drops to 0 (no overlap between consecutive frames)."""
    return [frame(start + i, clip, boxes=make_boxes(shift=(0.0, 0.25 * (i % 2)))) for i in range(n)]


def levels(seq, per=5, clip="clipA"):
    """Frames whose consecutive-frame consistency is 1 (level 1) or 0 (level 0) for `per` transitions per window."""
    out, i = [frame(0, clip)], 0
    for lv in seq:
        for _ in range(per):
            i += 1
            out.append(frame(i, clip, boxes=make_boxes() if lv == 1 else make_boxes(shift=(0.0, 0.25 * (i % 2)))))
    return out


def test_no_fire_on_a_stable_stream():
    out = make({"step": dict(STEP_RULE)}).run(good(31))
    assert len(out["windows"]) == 6 and out["triggers"] == []


def test_fires_when_the_statistic_drops_by_more_than_delta():
    out = make({"step": dict(STEP_RULE)}).run(good(11) + bad(10, start=11))
    t = out["triggers"][0]
    assert t.drop > 0.2 and t.rule == "step" and t.stage == "screening" and t.signal == "consistency" and t.delta == 0.2
    assert t.statistic_after < t.statistic_before and t.comparator == "previous"


def test_n_consecutive_windows_are_required():
    frames = good(11) + bad(15, start=11)
    two = make({"sus": dict(SUST_RULE)}).run(frames)
    one = make({"sus": dict(SUST_RULE, n_consecutive=1)}).run(frames)
    assert two["triggers"][0].window_id == one["triggers"][0].window_id + 1


def test_cooldown_and_own_state_restart_after_a_fire():
    frames = good(11) + bad(41, start=11)
    out = make({"sus": dict(SUST_RULE, cooldown_windows=3)}).run(frames)
    fired = [t.window_id for t in out["triggers"]]
    assert 1 <= len(fired) <= 2 and all(b - a >= 4 for a, b in zip(fired, fired[1:]))   # after the fire the low level is the new anchor


def test_only_trigger_enabled_signals_can_fire():
    out = make({"step": dict(STEP_RULE)}, consistency={"measure": True, "trigger": False},
               confidence={"measure": True, "trigger": True}).run(good(11) + bad(10, start=11))
    assert out["triggers"] == []                      # confidence is stable, consistency is silent


def test_measure_false_computes_nothing():
    eng = make({"step": dict(STEP_RULE)}, confidence={"measure": False})
    out = eng.run(good(6))
    assert "confidence" not in {v.signal for v in out["values"]} and "confidence" not in eng.signals


def test_scene_cut_is_logged_windows_do_not_cross_but_state_does(tmp_path):
    eng = make({"step": dict(STEP_RULE)}, tmp_path)
    out = eng.run(good(8, "clipA") + good(11, "clipB"))
    assert [c["kind"] for c in out["context"]] == ["scene_cut"] and out["context"][0]["from_clip"] == "clipA"
    assert len(out["windows"]) == 3 and eng.n_pairs == 17 and eng.dropped_partial_windows >= 1
    assert out["windows"][1].reference == out["windows"][0].statistic        # 'previous' spans the cut
    assert (tmp_path / "context.jsonl").read_text().count("scene_cut") == 1


def test_frame_windows_align_with_pair_windows():
    out = make({"step": dict(STEP_RULE)}, confidence={"measure": True}).run(good(11))
    per = {}
    for w in out["windows"]:
        per.setdefault(w.signal, []).append(w)
    assert len(per["consistency"]) == len(per["confidence"]) == 2 and per["confidence"][0].n_items == 5


def test_invalid_values_are_logged_as_null_never_a_number(tmp_path):
    make({"step": dict(STEP_RULE)}, tmp_path).run([frame(i, boxes=np.zeros((0, 6))) for i in range(6)])
    rows = [json.loads(l) for l in open(tmp_path / "signals.jsonl")]
    assert rows and all(r["value"] is None and r["valid"] is False and r["invalid_reason"] == "EMPTY_PAIR" for r in rows)


# ------------------------------------------------------------------ independent rule states
def test_step_and_sustained_both_fire_in_the_same_stream():
    frames = levels([1, 1, 0, 0, 0, 0])
    both = make({"step": dict(STEP_RULE, cooldown_windows=0), "sustained": dict(SUST_RULE)}).run(frames)
    by_rule = {}
    for t in both["triggers"]:
        by_rule.setdefault(t.rule, []).append(t.window_id)
    assert by_rule["step"] == [2] and by_rule["sustained"] == [3]        # the step fire must not reset the sustained rule
    assert {t.stage for t in both["triggers"] if t.rule == "sustained"} == {"confirmation"}


def test_each_rule_behaves_exactly_as_when_it_runs_alone():
    frames = levels([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    rules = {"step": dict(STEP_RULE, cooldown_windows=1), "sustained": dict(SUST_RULE),
             "rate": {"type": "rate", "enabled": True, "stage": "confirmation", "flag_rule": "step", "windows": 6, "min_flagged": 2}}
    joint = make(rules).run(frames)["triggers"]
    for name in rules:
        alone_rules = {name: rules[name]} if name != "rate" else {"step": dict(rules["step"]), "rate": rules["rate"]}
        alone = make(alone_rules).run(frames)["triggers"]
        assert [t.window_id for t in alone if t.rule == name] == [t.window_id for t in joint if t.rule == name], name


def test_step_or_sustained_differs_from_step_alone():
    frames = levels([1, 1, 0, 0, 0, 0])
    step = make({"step": dict(STEP_RULE, cooldown_windows=0)}).run(frames)["triggers"]
    both = make({"step": dict(STEP_RULE, cooldown_windows=0), "sustained": dict(SUST_RULE)}).run(frames)["triggers"]
    assert len(both) > len(step)


# ------------------------------------------------------------------ rate rule
RATE = {"type": "rate", "enabled": True, "stage": "confirmation", "flag_rule": "step", "windows": 6, "min_flagged": 3}


def test_rate_rule_counts_flagged_windows_not_fired_requests():
    frames = levels([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])          # a drop every second window
    once = dict(STEP_RULE, cooldown_windows=100)              # the step rule fires once and then stays silent for 100 windows
    out = make({"step": once, "rate": RATE}).run(frames)
    assert [t.window_id for t in out["triggers"] if t.rule == "step"] == [1]
    rate = [t for t in out["triggers"] if t.rule == "rate"]
    assert rate and rate[0].window_id == 5 and rate[0].flag_rate["flagged"] == 3 and rate[0].stage == "confirmation"


def test_rate_rule_fires_again_only_after_k_new_flags():
    frames = levels([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    out = make({"step": dict(STEP_RULE, cooldown_windows=100), "rate": RATE}).run(frames)
    assert [t.window_id for t in out["triggers"] if t.rule == "rate"] == [5, 11]


def test_rate_rule_validation():
    sc = validate_signals(sig_cfg(), fps=6.0)
    step = dict(STEP_RULE)
    with pytest.raises(Exception, match="flag_rule"):
        validate_trigger({"trigger": {"rules": {"step": step, "r": dict(RATE, flag_rule="nope")}}}, sc)
    with pytest.raises(Exception, match="exceeds"):
        validate_trigger({"trigger": {"rules": {"step": step, "r": dict(RATE, min_flagged=7)}}}, sc)
    with pytest.raises(Exception, match="stage"):
        validate_trigger({"trigger": {"rules": {"step": {k: v for k, v in step.items() if k != "stage"}}}}, sc)
    validate_trigger({"trigger": {"rules": {"step": step, "r": dict(RATE, enabled=False, flag_rule="nope")}}}, sc)   # disabled: not checked


# ------------------------------------------------------------------ evidence-carrying requests
def healthy_bundle():
    h = np.linspace(0.0, 0.1, 10)
    return ReferenceBundle("healthy_changes", {"previous": h, "anchor": h}, {"signal": "consistency"})


def test_request_carries_evidence_and_context(tmp_path):
    frames = levels([1, 1, 1, 0, 1, 0])
    eng = make({"step": dict(STEP_RULE, cooldown_windows=0)}, tmp_path, healthy=healthy_bundle(), confidence={"measure": True},
               platform_motion={"measure": True})
    out = eng.run(frames)
    r = out["triggers"][0]
    assert r.request_id == "step-consistency-3" and r.stage == "screening" and r.rule_type == "relative_drop"
    assert r.statistic_before == pytest.approx(1.0) and r.statistic_after == pytest.approx(0.0) and r.drop == pytest.approx(1.0)
    assert r.p_value == pytest.approx(1 / 11) and r.p_value_note is None                    # (1 + #healthy >= drop) / (n + 1)
    h = healthy_bundle().arrays["previous"]
    assert r.p_value_floor == pytest.approx(1 / 11) and r.p_value == r.p_value_floor         # the drop exceeds every healthy value: p sits at its floor
    mad = float(np.median(np.abs(h - np.median(h))))
    assert r.robust_z == pytest.approx(1.0 / (1.4826 * mad)) and r.robust_z_note is None      # unlike p, robust_z keeps growing with the drop
    fr = r.flag_rate
    assert fr["windows"] == 4 and fr["flagged"] == 1 and fr["rate"] == 0.25 and fr["healthy_flag_probability"] == 0.0 and fr["binomial_p"] == 0.0
    assert set(r.context) == {"confidence", "platform_motion"}                              # every measured signal except the firing one
    assert set(r.context["confidence"]) >= {"window_statistic", "latest_value", "aggregator"}
    row = json.loads(open(tmp_path / "requests.jsonl").readline())
    assert row["kind"] == "adaptation_request" and row["stage"] == "screening" and "context" in row


def test_without_a_healthy_distribution_the_p_value_is_null_with_a_reason():
    r = make({"step": dict(STEP_RULE)}).run(good(11) + bad(10, start=11))["triggers"][0]
    assert r.p_value is None and r.p_value_note == "no_healthy_distribution"


def test_context_never_influences_the_decision():
    frames = good(11) + bad(10, start=11) + good(10, start=21) + bad(10, start=31)
    rules = {"step": dict(STEP_RULE, cooldown_windows=0), "sustained": dict(SUST_RULE), "rate": dict(RATE)}
    a = make(rules).run(frames)["triggers"]
    b = make(rules, confidence={"measure": True}, platform_motion={"measure": True}).run(frames)["triggers"]
    key = lambda ts: [(t.rule, t.window_id, t.drop, t.stage) for t in ts]   # noqa: E731
    assert key(a) == key(b) and a


def test_output_lines_follow_the_schema(tmp_path):
    schema = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "local_signal_event.json").read_text())
    eng = make({"step": dict(STEP_RULE, cooldown_windows=0)}, tmp_path, healthy=healthy_bundle(), confidence={"measure": True})
    eng.run(good(11, "A") + bad(10, "B", 11))
    kinds = {"signals": "SignalValue", "windows": "WindowStat", "requests": "AdaptationRequest", "context": "ContextEvent", "frames": "FrameLog"}
    n = 0
    for f, name in kinds.items():
        for line in open(tmp_path / f"{f}.jsonl"):
            row = json.loads(line)
            n += 1
            req = schema["definitions"][name]["required"]
            assert all(k in row for k in req), (f, [k for k in req if k not in row])
            if name == "SignalValue":
                assert (row["value"] is None) == (not row["valid"])
    assert n > 0


def test_robust_z_grows_with_the_drop_while_p_saturates_and_zero_mad_is_reported():
    from edge_signals.base import ReferenceBundle
    eng = make({"step": dict(STEP_RULE, cooldown_windows=0)}, healthy=healthy_bundle()).trigger
    z1, _ = eng.robust_z("previous", 0.5)
    z2, _ = eng.robust_z("previous", 1.0)
    assert z2 == pytest.approx(2 * z1) and eng.p_value("previous", 0.5)[0] == eng.p_value("previous", 1.0)[0] == eng.p_floor("previous")
    flat = ReferenceBundle("healthy_changes", {"previous": np.zeros(10), "anchor": np.zeros(10)}, {})
    trg_flat = make({"step": dict(STEP_RULE)}, healthy=flat).trigger
    assert trg_flat.robust_z("previous", 0.3) == (None, "zero_mad")
    trg_none = make({"step": dict(STEP_RULE)}).trigger
    assert trg_none.robust_z("previous", 0.3) == (None, "no_healthy_distribution") and trg_none.p_floor("previous") is None
