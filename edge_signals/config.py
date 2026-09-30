"""YAML configuration of the signals and the trigger, with load-time validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional

import yaml

from .registry import available
from .windows import AGGREGATORS, CHANGES, COMPARATORS

ALLOWED_KEYS = {"measure", "trigger", "window_transitions", "window_seconds", "aggregator", "comparator", "rolling_windows", "change",
                "min_valid_fraction", "params"}


class ConfigError(ValueError):
    pass


def load_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def validate_signals(cfg: dict, fps: Optional[float] = None) -> Dict[str, dict]:
    """Return {signal_name: fully resolved settings}. Raises ConfigError on any inconsistency.
    A window is set either by window_transitions or by window_seconds (converted with the stream frame rate `fps`)."""
    reg = available()
    defaults = cfg.get("defaults", {})
    out = {}
    if "signals" not in cfg or not cfg["signals"]:
        raise ConfigError("signals: at least one signal must be configured")
    for name, raw in cfg["signals"].items():
        if name not in reg:
            raise ConfigError(f"unknown signal '{name}'; available: {sorted(reg)}")
        bad = set(raw or {}) - ALLOWED_KEYS
        if bad:
            raise ConfigError(f"{name}: unknown keys {sorted(bad)}")
        s = copy.deepcopy(defaults)
        if "window_transitions" in defaults and "window_seconds" in defaults:
            raise ConfigError("defaults: give window_transitions or window_seconds, not both")
        own = raw or {}
        if "window_transitions" in own and "window_seconds" in own:
            raise ConfigError(f"{name}: give window_transitions or window_seconds, not both")
        if "window_seconds" in own:
            s.pop("window_transitions", None)
        if "window_transitions" in own:
            s.pop("window_seconds", None)
        s.update(own)
        if "window_seconds" in s:
            if not fps or fps <= 0:
                raise ConfigError(f"{name}: window_seconds needs the stream frame rate (configs/inference.yaml: stream.fps)")
            if not isinstance(s["window_seconds"], (int, float)) or s["window_seconds"] <= 0:
                raise ConfigError(f"{name}: window_seconds must be a number > 0")
            s["window_transitions"] = int(round(s["window_seconds"] * fps))     # one transition = one frame interval
            if s["window_transitions"] < 1:
                raise ConfigError(f"{name}: window_seconds x fps rounds to less than one transition")
        s.setdefault("measure", False)
        s.setdefault("trigger", False)
        s.setdefault("params", {})
        if not isinstance(s["measure"], bool) or not isinstance(s["trigger"], bool):
            raise ConfigError(f"{name}: measure and trigger must be true/false")
        if s["trigger"] and not s["measure"]:
            raise ConfigError(f"{name}: trigger: true requires measure: true")
        for k, allowed in (("aggregator", AGGREGATORS), ("comparator", COMPARATORS), ("change", CHANGES)):
            if s.get(k) not in allowed:
                raise ConfigError(f"{name}: {k} must be one of {allowed} (got {s.get(k)!r})")
        if not isinstance(s.get("window_transitions"), int) or s["window_transitions"] < 1:
            raise ConfigError(f"{name}: window_transitions must be an integer >= 1")
        if s["comparator"] == "rolling":
            if not isinstance(s.get("rolling_windows"), int) or s["rolling_windows"] < 1:
                raise ConfigError(f"{name}: comparator rolling needs rolling_windows: M (integer >= 1)")
        if s["comparator"] == "reference" and reg[name].family != "drift":
            raise ConfigError(f"{name}: comparator 'reference' is only defined for drift signals")
        if not 0 < float(s.get("min_valid_fraction", 0.5)) <= 1:
            raise ConfigError(f"{name}: min_valid_fraction must be in (0, 1]")
        out[name] = s
    return out


def validate_trigger(cfg: dict, signals: Dict[str, dict]) -> dict:
    t = copy.deepcopy(cfg.get("trigger", {}) or {})
    rules = t.get("rules") or {}
    enabled = {n: rc for n, rc in rules.items() if rc.get("enabled", False)}
    for name, rc in enabled.items():
        typ = rc.get("type", name)
        if typ not in ("relative_drop", "rate"):
            continue                                    # a hook: TriggerEngine refuses it with a clear message
        if rc.get("stage") not in ("screening", "confirmation"):
            raise ConfigError(f"trigger.rules.{name}.stage must be screening | confirmation")
        if not any(s["trigger"] for s in signals.values()):
            raise ConfigError(f"trigger.rules.{name} is enabled but no signal has trigger: true")
        if typ == "relative_drop":
            for k in ("delta", "n_consecutive", "cooldown_windows", "comparator"):
                if k not in rc:
                    raise ConfigError(f"trigger.rules.{name}.{k} is required")
            if not isinstance(rc["delta"], (int, float)) or rc["delta"] <= 0:
                raise ConfigError(f"trigger.rules.{name}.delta must be a number > 0")
            if not isinstance(rc["n_consecutive"], int) or rc["n_consecutive"] < 1:
                raise ConfigError(f"trigger.rules.{name}.n_consecutive must be an integer >= 1")
            if not isinstance(rc["cooldown_windows"], int) or rc["cooldown_windows"] < 0:
                raise ConfigError(f"trigger.rules.{name}.cooldown_windows must be an integer >= 0")
            if rc["comparator"] not in ("previous", "anchor", "rolling"):
                raise ConfigError(f"trigger.rules.{name}.comparator must be previous | anchor | rolling")
            if rc["comparator"] == "rolling" and not (isinstance(rc.get("rolling_windows"), int) and rc["rolling_windows"] >= 1):
                raise ConfigError(f"trigger.rules.{name}: comparator rolling needs rolling_windows: M")
        else:
            src = enabled.get(rc.get("flag_rule"))
            if src is None or src.get("type", rc.get("flag_rule")) != "relative_drop":
                raise ConfigError(f"trigger.rules.{name}.flag_rule must name an enabled relative_drop rule")
            for k in ("windows", "min_flagged"):
                if not isinstance(rc.get(k), int) or rc[k] < 1:
                    raise ConfigError(f"trigger.rules.{name}.{k} must be an integer >= 1")
            if rc["min_flagged"] > rc["windows"]:
                raise ConfigError(f"trigger.rules.{name}: min_flagged ({rc['min_flagged']}) exceeds windows ({rc['windows']})")
            if not isinstance(rc.get("cooldown_windows", 0), int) or rc.get("cooldown_windows", 0) < 0:
                raise ConfigError(f"trigger.rules.{name}.cooldown_windows must be an integer >= 0")
    ev = t.get("evidence") or {}
    if ev and (not isinstance(ev.get("windows", 12), int) or ev.get("windows", 12) < 1):
        raise ConfigError("trigger.evidence.windows must be an integer >= 1")
    return t


def load_configs(signals_path, trigger_path, fps: Optional[float] = None):
    sc = validate_signals(load_yaml(signals_path), fps)
    return sc, validate_trigger(load_yaml(trigger_path), sc)
