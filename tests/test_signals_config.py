import copy

import pytest

from conftest import sig_cfg
from edge_signals.config import ConfigError, validate_signals, validate_trigger
from edge_signals.trigger import TriggerEngine


def test_valid_config_resolves_defaults():
    s = validate_signals(sig_cfg(confidence={"measure": True}))
    assert s["consistency"]["window_transitions"] == 5 and s["confidence"]["trigger"] is False


def test_trigger_requires_measure():
    with pytest.raises(ConfigError, match="requires measure"):
        validate_signals(sig_cfg(confidence={"measure": False, "trigger": True}))


@pytest.mark.parametrize("key,val", [("aggregator", "max"), ("comparator", "last"), ("change", "diff"), ("window_transitions", 0)])
def test_bad_enumerations_are_rejected(key, val):
    c = sig_cfg()
    c["signals"]["consistency"][key] = val
    with pytest.raises(ConfigError):
        validate_signals(c)


def test_rolling_needs_M_and_unknown_signal_and_keys():
    c = sig_cfg()
    c["signals"]["consistency"].update(comparator="rolling")
    c["defaults"].pop("rolling_windows")
    with pytest.raises(ConfigError, match="rolling_windows"):
        validate_signals(c)
    with pytest.raises(ConfigError, match="unknown signal"):
        validate_signals(sig_cfg(nonsense={"measure": True}))
    c = sig_cfg()
    c["signals"]["consistency"]["treshold"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        validate_signals(c)


def test_reference_comparator_only_for_drift():
    c = sig_cfg()
    c["signals"]["consistency"]["comparator"] = "reference"
    with pytest.raises(ConfigError, match="drift"):
        validate_signals(c)


def test_trigger_config_validation(trig_cfg):
    s = validate_signals(sig_cfg())
    assert validate_trigger(trig_cfg, s)
    bad = copy.deepcopy(trig_cfg)
    bad["trigger"]["rules"]["step"]["delta"] = 0
    with pytest.raises(ConfigError, match="delta"):
        validate_trigger(bad, s)
    s2 = validate_signals(sig_cfg(consistency={"measure": True, "trigger": False}))
    with pytest.raises(ConfigError, match="no signal has trigger"):
        validate_trigger(trig_cfg, s2)


def test_prioritised_trigger_hooks_exist_but_refuse_to_run(trig_cfg):
    cfg = copy.deepcopy(trig_cfg)
    cfg["trigger"]["rules"]["periodic"] = {"enabled": True, "stage": "screening"}
    with pytest.raises(NotImplementedError, match="hook"):
        TriggerEngine(cfg["trigger"], ["consistency"])
