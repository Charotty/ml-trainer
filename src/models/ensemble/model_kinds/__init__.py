"""Plugin registry for extra ``model_kind`` values (Variant A stage 7).

Each model lives in its own module in this package and registers a factory
with :func:`register`. Modules are auto-imported by :func:`load_all`, so new
model kinds can land in separate branches/PRs without editing shared files.

Factory signature::

    factory(target_name: str, *, estimator_profile: str,
            feature_names: list[str], target_names: list[str],
            y_context: np.ndarray | None) -> sklearn-like regressor

``y_context`` is the full training target matrix (rows aligned with the
per-target training rows) so multi-output models can fit jointly.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

ModelFactory = Callable[..., Any]


@dataclass
class ModelKindSpec:
    name: str
    factory: ModelFactory
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


REGISTRY: dict[str, ModelKindSpec] = {}
ALIASES: dict[str, str] = {}
_LOADED = False


def register(name: str, *, aliases: tuple[str, ...] = (), description: str = ""):
    def _decorator(factory: ModelFactory) -> ModelFactory:
        key = name.strip().lower()
        REGISTRY[key] = ModelKindSpec(key, factory, tuple(aliases), description)
        ALIASES[key] = key
        for alias in aliases:
            ALIASES[alias.strip().lower()] = key
        return factory

    return _decorator


def load_all() -> None:
    global _LOADED
    if _LOADED:
        return
    for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{info.name}")
    _LOADED = True


def resolve(kind: str) -> str | None:
    load_all()
    return ALIASES.get(str(kind).strip().lower())


def is_registered(kind: str) -> bool:
    return resolve(kind) is not None


def build(kind: str, target_name: str, **ctx: Any) -> tuple[str, Any]:
    key = resolve(kind)
    if key is None:
        raise KeyError(f"model_kind {kind!r} is not registered")
    return key, REGISTRY[key].factory(target_name, **ctx)


def all_aliases() -> dict[str, str]:
    load_all()
    return dict(ALIASES)
