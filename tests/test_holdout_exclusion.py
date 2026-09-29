"""Holdout exclusion and parser guards for Variant A."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.holdout import (
    assert_no_holdout_leak,
    holdout_surname_keys,
    is_holdout_name,
    load_holdout_config,
    normalize_surname_key,
)
from src.data.xlsx_displacement_parser import (
    DEFAULT_XLSX_PATH,
    build_vybor_from_xlsx,
    parse_xlsx_raw_table,
)

ROOT = Path(__file__).resolve().parents[1]
XLSX_13 = next(iter(ROOT.glob("*конечное*-13*.xlsx")), None) or next(
    iter(ROOT.glob("*13*.xlsx")), None
)


def test_normalize_surname_key_yo_and_latin():
    assert normalize_surname_key("Крёмас") == normalize_surname_key("Кремас")
    assert normalize_surname_key("Krems Gennadiy") == "krems"
    assert normalize_surname_key("Somova Mariya") == "somova"
    assert is_holdout_name("Кремс")
    assert is_holdout_name("Vegerin")
    assert is_holdout_name("Zalesskaya")
    assert not is_holdout_name("Кириллин")


def test_holdout_config_lists_five():
    cfg = load_holdout_config()
    patients = cfg["patients"]
    assert len(patients) == 5
    surnames = {normalize_surname_key(p["surname"]) for p in patients}
    assert surnames == {
        normalize_surname_key("Вегерин"),
        normalize_surname_key("Залесская"),
        normalize_surname_key("Сомова"),
        normalize_surname_key("Шилова"),
        normalize_surname_key("Кремас"),
    }
    keys = holdout_surname_keys(cfg)
    assert "krems" in keys
    assert "kremas" in keys


@pytest.mark.skipif(XLSX_13 is None or not Path(XLSX_13).exists(), reason="-13 xlsx missing")
def test_parser_skips_repeated_header_and_holdout():
    raw_all = parse_xlsx_raw_table(XLSX_13, exclude_surnames=[])
    assert "ФИО" not in set(raw_all["fio"].astype(str))
    # No colliding sheet № reuse as row_no identity.
    assert raw_all["row_no"].is_unique
    # Holdout trio present when not excluded.
    names = set(raw_all["fio"].astype(str).str.lower())
    assert any("крема" in n or "кремс" in n for n in names)
    assert any("сомова" in n for n in names)
    assert any("вегерин" in n for n in names)

    raw_ex = parse_xlsx_raw_table(XLSX_13, exclude_surnames=holdout_surname_keys())
    names_ex = set(raw_ex["fio"].map(normalize_surname_key))
    for banned in holdout_surname_keys():
        assert banned not in names_ex


@pytest.mark.skipif(XLSX_13 is None or not Path(XLSX_13).exists(), reason="-13 xlsx missing")
def test_build_vybor_excludes_holdout_and_sets_data_origin():
    df = build_vybor_from_xlsx(XLSX_13, boku_path=None)
    assert_no_holdout_leak(df)
    assert str(df["data_origin"].iloc[0]) == Path(XLSX_13).name
    assert len(df) >= 80
    # case_id uniqueness after excel_row fix
    assert df["case_id"].is_unique


@pytest.mark.skipif(XLSX_13 is None or not Path(XLSX_13).exists(), reason="-13 xlsx missing")
def test_assert_no_holdout_leak_raises_when_present():
    df = build_vybor_from_xlsx(XLSX_13, boku_path=None, exclude_surnames=[])
    with pytest.raises(ValueError, match="Holdout leakage"):
        assert_no_holdout_leak(df)
