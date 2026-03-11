
from __future__ import annotations

import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform with clipping."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid."""
    return 1 / (1 + np.exp(-x))
