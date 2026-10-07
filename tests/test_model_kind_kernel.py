import pytest

from tests.model_kind_helpers import run_kind


@pytest.mark.parametrize("kind", ["gpr", "svr_linear", "svr_rbf", "knn"])
def test_kernel_kinds_nested_cv(kind):
    run_kind(kind)
