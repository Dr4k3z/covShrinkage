from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import fast_logdet


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


class PrecisionNotFittedError(Exception):
    """
    Exception raised when the precision matrix has not been computed yet,
    but the user calls a method that assumes it's existence.
    """

    def __init__(self) -> None:
        self._msg = "Precision matrix is not computed. You have either not fitted the model yet or set stop_precision to True."
        super().__init__(self._msg)


class MatrixTypeComparisonError(Exception):
    """
    Exception raised when the other input is neither a numpy array nor a ShrunkedCovariance instance.
    """

    def __init__(self) -> None:
        self._msg = "The other input is neither a numpy array nor a ShrunkedCovariance instance."
        super().__init__(self._msg)


class MatrixShapeComparisonError(Exception):
    """
    Exception raised when the other input does not have the same shape as the current covariance matrix.
    """

    def __init__(self) -> None:
        self._msg = "The other input does not have the same shape as the current covariance matrix."
        super().__init__(self._msg)


class UndefinedNormError(Exception):
    """
    Exception raised when the specified norm is not implemented.
    """

    def __init__(self) -> None:
        self._msg = "The specified norm is not implemented."
        super().__init__(self._msg)


def log_likelihood(cov: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute the sample mean of the log-likelihood under the covariance model.
    """
    p = precision.shape[0]

    log_likelihood_: float = (
        -np.sum(cov * precision) + fast_logdet(precision) - p * np.log(2 * np.pi)
    )
    log_likelihood_ /= 2.0

    return log_likelihood_


def validate_data(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    This decorator is used to check if the input data is indeed a numpy array.
    If not, it raises a NotNumpyArrayError. In the future I might also add more checks
    and their corresponding exceptions.
    """

    def inner(*args: Any, **kwargs: Any) -> Any:
        if args:
            X = args[1] if len(args) > 1 else args[0]
        else:
            X = kwargs.get("X", None)
        if not isinstance(X, np.ndarray):
            raise NotNumpyArrayError()
        return func(*args, **kwargs)

    return inner


norms_dictionary: dict[str, Callable[[np.ndarray], float]] = {
    "frobenius": lambda x: np.sum(x**2),
    "spectral": lambda x: np.amax(np.linalg.svd(np.dot(x.T, x), compute_uv=False)),
}


class ShrunkedCovariance(BaseEstimator, ABC):
    """
    This is an abstract base class for all shrinkage methods.
    It inherits from scikit-learn BaseEstimator, so it can be used in scikit-learn pipelines.
    It provides a common interface for fitting and retrieving the shrunk covariance matrix.
    """

    __array_priority__ = 10.0  # NumPy will call our __rsub__ method instead of its own

    def __init__(self, stop_precision: bool = True, assume_centered: bool = False) -> None:
        self._stop_precision = stop_precision
        self._assume_centered = assume_centered

        self._covariance: np.ndarray | None = None
        self._precision: np.ndarray | None = None

    @abstractmethod
    def _fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        pass

    @validate_data
    def fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> ShrunkedCovariance:
        """
        Common interface to fit the covariance matrix to the data.
        This function relies on the _fit method, which is implemented in the subclasses.
        It performs data validation and centering if required. Also, if stop_precision is
        set to True, it computes the pseudo-inverse of the covariance matrix to get the precision
        matrix. Returns the instance itself.
        """
        if not self._assume_centered:
            X = X - np.mean(X, axis=0)

        covariance = self._fit(X, *args, **kwargs)
        self._covariance = covariance

        if self._stop_precision:
            self._precision = np.linalg.pinv(covariance)
        else:
            self._precision = None

        return self

    @validate_data
    def score(self, X_test: np.ndarray) -> float:
        """
        Compute the log-likelihood fo X_test under the estimated covariance model.
        This function requires that the model has been fitted and that the precision matrix
        has been computed (i.e., stop_precision is False). It also performs data validation
        and centering if required. Returns the log-likelihood value.
        """
        if self._assume_centered:
            X_test = X_test - np.mean(X_test, axis=0)

        test_cov = self._fit(X_test)
        res = log_likelihood(test_cov, self.precision)

        return res

    def error_norm(
        self,
        other: np.ndarray | ShrunkedCovariance,
        norm: Literal["frobenius", "spectral"] = "frobenius",
        scaling: bool = True,
        squared: bool = True,
    ) -> float:
        """
        Compute the error norm between the estimated covariance matrix and another
        covariance matrix. The other matrix can be either a numpy array or another
        ShrunkedCovariance instance. The norm can be either "frobenius" or "spectral".
        If scaling is True, the result is divided by the number of features. If squared is True,
        the squared norm is returned, otherwise the square root of the norm is returned.
        """
        if norm not in norms_dictionary:
            raise UndefinedNormError()

        if isinstance(other, ShrunkedCovariance):
            if other.shape != self.shape:
                raise MatrixShapeComparisonError()

            error = other.covariance - self.covariance
        elif isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise MatrixShapeComparisonError()

            error = other - self.covariance
        else:
            raise MatrixTypeComparisonError()

        squared_norm = norms_dictionary[norm](error)
        if scaling:
            squared_norm /= error.shape[0]
        return squared_norm if squared else float(np.sqrt(squared_norm))

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise CovarianceNotFittedError()

        return self._covariance

    @property
    def precision(self) -> np.ndarray:
        if self._precision is None:
            raise PrecisionNotFittedError()

        return self._precision

    @property
    def shape(self) -> tuple[int, int]:
        if self._covariance is None:
            raise CovarianceNotFittedError()

        return self._covariance.shape

    def _coerce_to_np_array(self, other: np.ndarray | ShrunkedCovariance) -> np.ndarray:
        if isinstance(other, np.ndarray):
            return other
        if isinstance(other, ShrunkedCovariance):
            return other.covariance
        raise TypeError("Trying to coerce an unsupported type to a numpy array.")

    def __add__(self, other: np.ndarray | ShrunkedCovariance) -> np.ndarray:
        other = self._coerce_to_np_array(other)
        res: np.ndarray = np.array([])

        try:
            if other.shape != self.shape:
                raise MatrixShapeComparisonError()
        except AttributeError:
            raise MatrixTypeComparisonError() from None

        if isinstance(other, ShrunkedCovariance):
            res = self.covariance + other.covariance
        elif isinstance(other, np.ndarray):
            res = self.covariance + other
        else:
            raise MatrixTypeComparisonError()

        return res

    def __radd__(self, other: np.ndarray | ShrunkedCovariance) -> np.ndarray:
        return self.__add__(other)

    def __sub__(self, other: np.ndarray | ShrunkedCovariance) -> np.ndarray:
        other = self._coerce_to_np_array(other)
        res: np.ndarray = np.array([])

        try:
            if other.shape != self.shape:
                raise MatrixShapeComparisonError()
        except AttributeError:
            raise MatrixTypeComparisonError() from None

        if isinstance(other, ShrunkedCovariance):
            res = self.covariance - other.covariance
        elif isinstance(other, np.ndarray):
            res = self.covariance - other
        else:
            raise MatrixTypeComparisonError()

        return res

    def __rsub__(self, other: np.ndarray | ShrunkedCovariance) -> np.ndarray:
        res: np.ndarray = np.array([])

        try:
            if other.shape != self.shape:
                raise MatrixShapeComparisonError()
        except AttributeError:
            raise MatrixTypeComparisonError() from None

        if isinstance(other, ShrunkedCovariance):
            res = other.covariance - self.covariance
        elif isinstance(other, np.ndarray):
            res = other - self.covariance
        else:
            raise MatrixTypeComparisonError()

        return res

    def __array__(self, dtype: type | None = None) -> np.ndarray:
        # alert the user
        warnings.warn(
            "The instance {self.__class__.__name__} is being converted to a numpy array.",
            UserWarning,
            stacklevel=2,
        )
        return np.asanyarray(self.covariance, dtype=dtype)


class EmpiricalCovariance(ShrunkedCovariance):
    def __init__(self, stop_precision: bool = True, assume_centered: bool = True) -> None:
        super().__init__(stop_precision=stop_precision, assume_centered=assume_centered)

    def _fit(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        covariance: np.ndarray = np.dot(X.T, X) / n_samples
        return covariance
