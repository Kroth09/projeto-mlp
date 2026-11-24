"""Tipos compartilhados pelos módulos matemáticos."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float64]
"""Array NumPy de valores ``float64``."""

