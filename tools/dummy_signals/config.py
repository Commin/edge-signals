"""Configuration: default_config.yaml + user overrides; {range: [lo, hi]} values are drawn once per seed (counter-based, recorded)."""
from __future__ import annotations

import copy
import re
import zlib
from pathlib import Path

import yaml

from . import rng

HERE = Path(__file__).resolve().parent


class ConfigError(ValueError):
    pass


def deep_merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base[k], v) if k in base else v
        return out
    return copy.deepcopy(over)


def check_keys(base, over, path=""):
    """Unknown keys are errors (a typo must not silently change nothing)."""
    if isinstance(base, dict) and isinstance(over, dict):
        for k, v in over.items():
            if k not in base:
                raise ConfigError(f"unknown parameter '{path}{k}'")
            if isinstance(v, dict) and set(v) == {"range"}:
                continue
            check_keys(base[k], v, f"{path}{k}.")


UNITS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "d": 86400, None: 1}


def parse_duration(x) -> float:
    """Seconds as a number, or a number with a unit: 30s, 40m, 40min, 2h, 1d (units can be mixed freely in one file)."""
    if isinstance(x, bool):
        raise ConfigError(f"duration '{x}': use seconds or 30s / 40m / 40min / 2h / 1d")
    if isinstance(x, (int, float)):
        return float(x)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*(sec|s|min|m|h|d)?\s*", str(x))
    if not m:
        raise ConfigError(f"duration '{x}': use seconds or 30s / 40m / 40min / 2h / 1d")
    return float(m.group(1)) * UNITS[m.group(2)]


def parse_switch(x, name):
    """on / off / true / false (YAML reads a bare off as false and on as true)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("on", "true", "yes"):
        return True
    if isinstance(x, str) and x.strip().lower() in ("off", "false", "no"):
        return False
    raise ConfigError(f"{name}: use on / off / true / false, got {x!r}")


PER_AREA = {"area_params.difficulty", "area_params.lambda_obj", "area_params.lambda_floor"}        # a range here is drawn once per area (by the World), not once per seed


def letter_names(k):
    return [chr(ord("A") + i) if i < 26 else f"A{i}" for i in range(k)]


def area_names(cfg):
    """Names of the areas: the `areas` list (names or {name, difficulty}), or A, B, C, ... (simple.n_areas of them)."""
    a = cfg.get("areas")
    if a is None:
        return letter_names(int(cfg["simple"]["n_areas"]))
    if not isinstance(a, list) or not a:
        raise ConfigError("areas: a non-empty list of names or {name, difficulty}")
    names = []
    for x in a:
        n = x.get("name") if isinstance(x, dict) else x
        if n is None or isinstance(n, bool) or isinstance(n, (list, dict)):
            raise ConfigError(f"areas: bad entry {x!r}")
        if isinstance(x, dict) and set(x) - {"name", "difficulty"}:
            raise ConfigError(f"areas: unknown key in {x!r} (use name, difficulty)")
        names.append(str(n))
    if len(set(names)) != len(names):
        raise ConfigError("areas: duplicate area name")
    return names


def area_difficulty(cfg):
    """{index: difficulty} for the areas that give one in the `areas` list."""
    out = {}
    for k, x in enumerate(cfg.get("areas") or []):
        if isinstance(x, dict) and x.get("difficulty") is not None:
            d = float(x["difficulty"])
            if not 0.0 <= d <= 1.0:
                raise ConfigError(f"area {x.get('name')}: difficulty must be in [0, 1]")
            out[k] = d
    return out


def area_index(names, x, where="area"):
    """Name (or index) -> index."""
    if not isinstance(x, bool) and str(x) in names:
        return names.index(str(x))
    if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < len(names):
        return x
    raise ConfigError(f"{where}: unknown area {x!r} (areas: {', '.join(names)})")


def resolve_route(route, names, duration_s, edge_id):
    """[{stay: A, for: 40m}, {move_to: B, over: 20m}, ...] -> contiguous steps with t0, t1 in seconds and area indices; the lengths must add up to the duration."""
    if not isinstance(route, list) or not route:
        raise ConfigError(f"edge {edge_id}: route must be a non-empty list")
    steps, t, cur = [], 0.0, None
    for i, st in enumerate(route):
        if not isinstance(st, dict):
            raise ConfigError(f"edge {edge_id}, route step {i + 1}: expected {{stay: AREA, for: DURATION}} or {{move_to: AREA, over: DURATION}}")
        keys = set(st)
        if keys == {"stay", "for"}:
            kind, area, d = "stay", area_index(names, st["stay"], f"edge {edge_id}, route step {i + 1}"), parse_duration(st["for"])
        elif keys == {"move_to", "over"}:
            kind, area, d = "move", area_index(names, st["move_to"], f"edge {edge_id}, route step {i + 1}"), parse_duration(st["over"])
        else:
            raise ConfigError(f"edge {edge_id}, route step {i + 1}: expected {{stay: AREA, for: DURATION}} or {{move_to: AREA, over: DURATION}}, got keys {sorted(keys)}")
        if d <= 0:
            raise ConfigError(f"edge {edge_id}, route step {i + 1}: the duration must be positive")
        if kind == "move":
            if cur is None:
                raise ConfigError(f"edge {edge_id}: a route must start with 'stay' (a move needs a previous area)")
            if area == cur:
                raise ConfigError(f"edge {edge_id}, route step {i + 1}: move_to the area the edge is already in")
        step = {"kind": kind, "t0": t, "t1": t + d, "area": area}
        if kind == "move":
            step["from_area"] = cur
        steps.append(step)
        t, cur = t + d, area
    if abs(t - duration_s) > 1e-6:
        raise ConfigError(f"edge {edge_id}: the route lasts {t:g} s but simple.duration is {duration_s:g} s (the steps must add up to the duration)")
    return steps


def resolve_edges(cfg, names, duration_s):
    """The edges of the world: id, mode (route | random), models stored on the edge, initial model, resolved route steps."""
    model_ids = [m["id"] for m in cfg["models"]]
    edges = list(cfg["edges"])
    n = cfg["simple"]["n_edges"]
    if n is not None:
        edges = edges[:int(n)] + [{"id": f"e{i + 1}"} for i in range(len(edges), int(n))]
    if not edges:
        raise ConfigError("at least one edge is needed")
    out, seen = [], set()
    for e in edges:
        if not isinstance(e, dict) or "id" not in e:
            raise ConfigError(f"edges: every edge needs an id, got {e!r}")
        if set(e) - {"id", "position", "models", "initial_model", "route"}:
            raise ConfigError(f"edge {e['id']}: unknown key(s) {sorted(set(e) - {'id', 'position', 'models', 'initial_model', 'route'})}")
        eid = str(e["id"])
        if eid in seen:
            raise ConfigError(f"duplicate edge id '{eid}'")
        seen.add(eid)
        mods = e.get("models")
        mods = list(model_ids) if mods is None else list(mods)
        for m in mods:
            if m not in model_ids:
                raise ConfigError(f"edge {eid}: unknown model '{m}' in models")
        init = e.get("initial_model", mods[0] if mods else None)
        if init not in model_ids:
            raise ConfigError(f"edge {eid}: unknown initial_model '{init}'")
        if init not in mods:
            raise ConfigError(f"edge {eid}: initial_model '{init}' is not among the models stored on the edge ({', '.join(mods)})")
        route = e.get("route")
        steps = resolve_route(route, names, duration_s, eid) if route is not None else None
        out.append({"id": eid, "mode": "route" if route is not None else "random", "position": e.get("position"), "models": mods, "initial_model": init, "route": route, "steps": steps})
    return out


def resolve(node, seed, path="", draws=None):
    """Replace every {range: [lo, hi]} by a number drawn from (seed, parameter path)."""
    if path in PER_AREA:
        return node
    if isinstance(node, dict):
        if set(node) == {"range"}:
            lo, hi = node["range"]
            v = lo + (hi - lo) * float(rng.uniform(seed, rng.L_CONFIG, zlib.crc32(path.encode())))
            if draws is not None:
                draws[path] = {"range": [lo, hi], "value": v}
            return v
        return {k: resolve(v, seed, f"{path}.{k}" if path else k, draws) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve(v, seed, f"{path}[{i}]", draws) for i, v in enumerate(node)]
    return node


def load(config=None, seed=0):
    """-> (resolved config, draws of the ranges)."""
    base = yaml.safe_load(open(HERE / "default_config.yaml"))
    if config is None:
        user = {}
    elif isinstance(config, (str, Path)):
        user = yaml.safe_load(open(config)) or {}
    else:
        user = copy.deepcopy(config)
    check_keys(base, user)
    cfg = deep_merge(base, user)
    draws = {}
    cfg = resolve(cfg, seed, "", draws)
    cfg["simple"]["duration_s"] = parse_duration(cfg["simple"]["duration"])
    cfg["random_episodes"] = parse_switch(cfg["random_episodes"], "random_episodes")
    names = area_names(cfg)
    K = len(names)
    cfg["simple"]["n_areas"] = K
    area_difficulty(cfg)
    if K < 1 or (cfg["simple"]["n_edges"] is not None and int(cfg["simple"]["n_edges"]) < 1):
        raise ConfigError("simple.n_areas and simple.n_edges must be >= 1")
    if not 0.0 <= cfg["simple"]["steepness"] <= 1.0:
        raise ConfigError("simple.steepness must be in [0, 1]")
    if not cfg["simple"]["perf_min"] < cfg["simple"]["perf_max"] <= 1.0:
        raise ConfigError("simple.perf_min must be below simple.perf_max (<= 1)")
    for m in cfg["models"]:
        if m["size"] not in ("s", "x"):
            raise ConfigError(f"model {m['id']}: size must be s or x")
        if set(m) - {"id", "size", "version", "trained_areas"}:
            raise ConfigError(f"model {m['id']}: unknown key(s) {sorted(set(m) - {'id', 'size', 'version', 'trained_areas'})}")
        for a in m["trained_areas"]:
            area_index(names, a, f"model {m['id']}, trained_areas")
    if len({m["id"] for m in cfg["models"]}) != len(cfg["models"]):
        raise ConfigError("duplicate model id")
    resolve_edges(cfg, names, cfg["simple"]["duration_s"])
    return cfg, draws
