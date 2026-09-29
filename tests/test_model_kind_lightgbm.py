import pytest

pytest.importorskip("lightgbm")

from tests.model_kind_helpers import run_kind  # noqa: E402


def test_lightgbm_kind_nested_cv():
    run_kind("lightgbm")
