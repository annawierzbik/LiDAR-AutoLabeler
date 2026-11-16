"""The guard that keeps GPU tests from running unless they are asked for by name."""
import pytest

from tests.conftest import gpu_requested


@pytest.mark.parametrize(("expr", "runs"), [
    ("gpu", True), ("gpu and worker", True), ("worker and gpu", True), ("(gpu)", True),
    ("worker", False), ("postgres", False), ("", False),
    ("not gpu and not e2e and not postgres and not worker", False), ("worker or not gpu", False),
    ("gpus", False), ("nogpu", False),
])
def test_gpu_tests_need_the_gpu_marker_by_name(expr, runs):
    assert gpu_requested(expr) is runs
