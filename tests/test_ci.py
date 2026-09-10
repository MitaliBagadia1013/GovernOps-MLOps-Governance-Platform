import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    import pandas
    import numpy
    import sklearn

    assert True


def test_python_version():
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 9


def test_directories():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for dir_name in ["models", "model_cards", "mlruns"]:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    assert True


def test_sanity():
    assert 1 + 1 == 2
    assert True
