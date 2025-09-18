from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class CovarianceNotFittedError(Exception):
    """
    Exception raised when the covariance matrix has not been computed yet,
    but the user calls a method that assumes it's existence.
    """

    def __init__(self) -> None:
        self._msg = "Covariance matrix is not computed. Please fit the model first."
        super().__init__(self._msg)


class NotNumpyArrayError(Exception):
    """
    Unless a few rare cases, all input data will usually be numpy arrays.
    This exception is raised when the input data is not a numpy array.
    """

    def __init__(self) -> None:
        self._msg = "Input data must be a numpy array."
        super().__init__(self._msg)


class ShrunkedCovariance(BaseEstimator, ABC):
    """
    This is an abstract base class for all shrinkage methods.
    It inherits from scikit-learn BaseEstimator, so it can be used in scikit-learn pipelines.
    It provides a common interface for fitting and retrieving the shrunk covariance matrix.
    """

    def __init__(self, stop_precision: bool = True, assume_centered: bool = True) -> None:
        self._stop_precision = stop_precision
        self._assume_centered = assume_centered

        self._covariance: np.ndarray | None = None

    @abstractmethod
    def _fit(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray) -> "ShrunkedCovariance":
        if not isinstance(X, np.ndarray):
            raise NotNumpyArrayError()

        if not self._assume_centered:
            X = X - np.mean(X, axis=0)

        covariance = self._fit(X)
        self._covariance = covariance

        return self

    @property
    def covariance_(self) -> np.ndarray:
        if self._covariance is None:
            raise CovarianceNotFittedError()

        return self._covariance

    @property
    def shape(self) -> tuple[int, int]:
        if self._covariance is None:
            raise CovarianceNotFittedError()
        return self._covariance.shape


class LedoitWolfShrinkage(ShrunkedCovariance):
    def __init__(self, stop_precision: bool = True, assume_centered: bool = True) -> None:
        super().__init__(stop_precision=stop_precision, assume_centered=assume_centered)

    def _fit(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[1], X.shape[1]))  # Placeholder implementation
