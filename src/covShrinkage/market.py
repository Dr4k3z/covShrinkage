from __future__ import annotations

import numpy as np

from .base import ShrunkedCovariance


class MarketShrinkage(ShrunkedCovariance):
    def __init__(self) -> None:
        super().__init__()

    def _fit(self, X: np.ndarray) -> np.ndarray:
        return np.array([])
