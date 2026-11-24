import sys

import numpy as np
import pandas as pd


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 12)


def test_numpy_is_available() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert values.mean() == 2.0


def test_pandas_is_available() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    assert frame["value"].mean() == 2.0

