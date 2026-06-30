"""Leakage-safe feature filtering for displacement models."""

from __future__ import annotations

LEAKY_FEATURE_SUBSTRINGS: tuple[str, ...] = ("delta_span", "lateral", "proj_diff_")


def is_leakage_feature(name: str) -> bool:
    """True when a column encodes lateral-scan or span-delta information."""
    return any(sub in name for sub in LEAKY_FEATURE_SUBSTRINGS)


def filter_model_features(names: list[str]) -> list[str]:
    return [n for n in names if not is_leakage_feature(n)]
