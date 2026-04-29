"""String-keyed factory registry for datasets, encoders, baselines, predictors, partitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

REGISTRY: dict[str, dict[str, Callable[..., Any]]] = defaultdict(dict)


def register(kind: str, name: str) -> Callable[[F], F]:
    """Decorator: register a factory under (kind, name)."""

    def wrap(fn: F) -> F:
        if name in REGISTRY[kind]:
            raise KeyError(f"{kind}/{name} already registered")
        REGISTRY[kind][name] = fn
        return fn

    return wrap


def build(kind: str, name: str, **cfg: Any) -> Any:
    """Look up (kind, name) in the registry and call the factory with cfg."""
    if kind not in REGISTRY or name not in REGISTRY[kind]:
        available = sorted(REGISTRY.get(kind, {}).keys())
        raise KeyError(f"{kind}/{name} not registered. Available {kind}: {available}")
    return REGISTRY[kind][name](**cfg)


def list_registered(kind: str) -> list[str]:
    return sorted(REGISTRY.get(kind, {}).keys())
