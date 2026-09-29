import pytest

from tests.model_kind_helpers import run_kind


@pytest.mark.parametrize("kind", ["mtenet", "chain", "joint_z"])
def test_multioutput_kinds_nested_cv(kind):
    run_kind(kind)
