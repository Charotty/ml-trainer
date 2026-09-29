import pytest

from tests.model_kind_helpers import run_kind


@pytest.mark.parametrize("kind", ["elasticnet", "huber_linear"])
def test_linear_kinds_nested_cv(kind):
    run_kind(kind)
