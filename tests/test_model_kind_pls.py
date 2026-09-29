import pytest

from tests.model_kind_helpers import run_kind


@pytest.mark.parametrize("kind", ["pls", "pls_multi"])
def test_pls_kinds_nested_cv(kind):
    run_kind(kind)
