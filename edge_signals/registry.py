"""Registry of signal plugins: one module per signal in edge_signals/builtin/, registered by name."""
from __future__ import annotations

from typing import Dict, Type

from .base import FAMILIES, GRANULARITIES, REQUIRES, Signal

_REGISTRY: Dict[str, Type[Signal]] = {}


def register(cls: Type[Signal]) -> Type[Signal]:
    if not cls.name:
        raise ValueError(f"{cls.__name__}: a signal needs a name")
    if cls.family not in FAMILIES or cls.granularity not in GRANULARITIES or not set(cls.requires) <= set(REQUIRES):
        raise ValueError(f"{cls.name}: invalid family/granularity/requires")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate signal name {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> Dict[str, Type[Signal]]:
    from . import builtin  # noqa: F401  (importing the package registers the built-in signals)
    return dict(_REGISTRY)


def get(name: str) -> Type[Signal]:
    reg = available()
    if name not in reg:
        raise KeyError(f"unknown signal '{name}'; available: {sorted(reg)}")
    return reg[name]
