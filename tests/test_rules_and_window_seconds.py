import pytest

from conftest import HOOKS, STEP_RULE, frame, sig_cfg
from edge_signals.config import ConfigError, validate_signals, validate_trigger
from edge_signals.engine import SignalEngine


def test_window_seconds_is_converted_with_the_frame_rate():
    s = validate_signals({"defaults": {"window_seconds": 5, "aggregator": "median", "comparator": "previous", "change": "difference"},
                          "signals": {"consistency": {"measure": True, "trigger": True}}}, fps=6.0)
    assert s["consistency"]["window_transitions"] == 30
    s = validate_signals({"defaults": {"window_transitions": 30, "aggregator": "median", "comparator": "previous", "change": "difference"},
                          "signals": {"consistency": {"measure": True, "window_seconds": 10}}}, fps=6.0)
    assert s["consistency"]["window_transitions"] == 60          # the signal's own setting replaces the inherited one


def test_window_seconds_needs_fps_and_excludes_window_transitions():
    c = {"defaults": {"aggregator": "median", "comparator": "previous", "change": "difference"},
         "signals": {"consistency": {"measure": True, "window_seconds": 5}}}
    with pytest.raises(ConfigError, match="frame rate"):
        validate_signals(c)
    c["signals"]["consistency"]["window_transitions"] = 30
    with pytest.raises(ConfigError, match="not both"):
        validate_signals(c, fps=6.0)


def test_window_records_its_length_in_seconds():
    sc = validate_signals(sig_cfg(), fps=6.0)
    out = SignalEngine(sc, validate_trigger({"trigger": {"rules": {}}}, sc), fps=6.0).run([frame(i) for i in range(7)])
    assert out["windows"][0].window_seconds == pytest.approx(5 / 6.0)


def test_trigger_rule_validation():
    sc = validate_signals(sig_cfg(), fps=6.0)
    for bad, msg in ((dict(STEP_RULE, comparator="reference"), "comparator"), (dict(STEP_RULE, comparator="rolling"), "rolling_windows"),
                     (dict(STEP_RULE, delta=-1), "delta"), ({k: v for k, v in STEP_RULE.items() if k != "n_consecutive"}, "n_consecutive"),
                     (dict(STEP_RULE, stage="both"), "stage")):
        with pytest.raises(ConfigError, match=msg):
            validate_trigger({"trigger": {"rules": {"x": bad}}}, sc)
    with pytest.raises(NotImplementedError):
        SignalEngine(sc, validate_trigger({"trigger": {"rules": {"p": {"type": "periodic", "enabled": True}}}}, sc))
    validate_trigger({"trigger": {"rules": {"p": {"type": "periodic", "enabled": False}, **HOOKS}}}, sc)         # disabled hooks are fine
