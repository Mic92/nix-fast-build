from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).parent.resolve()


@pytest.fixture(scope="session")
def test_root() -> Path:
    """
    Root directory of the tests
    """
    return TEST_ROOT
