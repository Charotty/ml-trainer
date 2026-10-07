"""Holdout patient helpers for Variant A (training exclusion + surname match)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT_PATH = REPO_ROOT / "config" / "holdout_patients.yaml"


def normalize_surname_key(name: object) -> str:
    """Lowercase surname token with ё→е and non-alnum stripped."""
    if name is None:
        return ""
    text = str(name).strip().lower().replace("ё", "е").replace("Ё", "е")
    token = re.split(r"[\s.]+", text)[0]
    return re.sub(r"[^a-zа-я0-9]", "", token)


def load_holdout_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_HOLDOUT_PATH
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "patients" not in raw:
        raise ValueError(f"Invalid holdout config: {cfg_path}")
    return dict(raw)


def holdout_surname_keys(config: Mapping[str, Any] | None = None) -> set[str]:
    cfg = config or load_holdout_config()
    keys: set[str] = set()
    for patient in cfg.get("patients") or []:
        keys.add(normalize_surname_key(patient.get("surname")))
        for alias in patient.get("aliases") or []:
            keys.add(normalize_surname_key(alias))
    keys.discard("")
    return keys


def holdout_case_ids(config: Mapping[str, Any] | None = None) -> set[str]:
    cfg = config or load_holdout_config()
    return {
        str(p["case_id"])
        for p in (cfg.get("patients") or [])
        if p.get("case_id")
    }


def is_holdout_name(name: object, banned_keys: Iterable[str] | None = None) -> bool:
    keys = set(banned_keys) if banned_keys is not None else holdout_surname_keys()
    return normalize_surname_key(name) in keys


def assert_no_holdout_leak(
    df,
    *,
    name_col: str | None = None,
    banned_keys: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
) -> None:
    """Raise if any holdout surname or case_id appears in a training frame."""
    keys = set(banned_keys) if banned_keys is not None else holdout_surname_keys()
    banned_cases = set(case_ids) if case_ids is not None else holdout_case_ids()
    col = name_col
    if col is None:
        for candidate in ("full_name", "fio", "case_id"):
            if candidate in df.columns:
                col = candidate
                break
    if col is None:
        raise ValueError("DataFrame needs full_name, fio, or case_id for holdout check")

    leaked_names: list[str] = []
    for value in df[col].astype(str):
        if col == "case_id":
            if value in banned_cases:
                leaked_names.append(value)
        elif is_holdout_name(value, keys):
            leaked_names.append(value)
    if "case_id" in df.columns and col != "case_id":
        for value in df["case_id"].astype(str):
            if value in banned_cases:
                leaked_names.append(value)
    if leaked_names:
        uniq = sorted(set(leaked_names))
        raise ValueError(
            f"Holdout leakage: {len(uniq)} banned patient(s) in training data: {uniq}"
        )


def mae_holdout_patients(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or load_holdout_config()
    return [p for p in (cfg.get("patients") or []) if p.get("score_mode") == "mae"]


def plausibility_holdout_patients(
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or load_holdout_config()
    return [
        p for p in (cfg.get("patients") or []) if p.get("score_mode") == "plausibility"
    ]
