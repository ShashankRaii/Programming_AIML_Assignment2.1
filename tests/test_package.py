"""testing the packages nd everything"""

import numcompute


def test_package_imports():
    assert numcompute is not None


def test_version_string():
    assert isinstance(numcompute.__version__, str)
    assert numcompute.__version__.count(".") == 2