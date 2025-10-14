from __future__ import annotations

from typing import Any

import numpy as np

from .base import MatrixShapeComparisonError, NotNumpyArrayError, ShrunkedCovariance


class TargetNotSetError(Exception):
    """
    Exception raised when the target matrix has not been set yet,
    but the user calls a method that assumes its existence.
    """

    def __init__(self) -> None:
        self._msg = "Target matrix is not set. Please set the target first."
        super().__init__(self._msg)


class CoeffInvalidError(Exception):
    """
    Exception raised when the shrinkage coefficients are invalid.
    """

    def __init__(self, coeff_name: str, coeff_value: Any) -> None:
        self._msg = f"Shrinkage coefficient {coeff_name} has invalid value {coeff_value}."
        super().__init__(self._msg)


class CoeffNotSetError(Exception):
    """
    Exception raised when the shrinkage coefficients have not been set yet,
    but the user calls a method that assumes their existence.
    """

    def __init__(self) -> None:
        self._msg = "Shrinkage coefficients are not set. Please fit the model first."
        super().__init__(self._msg)


class LinearShrinkage(ShrunkedCovariance):
    """
    Linear Shrinkage estimator for covariance matrices.
    The user of this class must set the shrinkage target and the shrinkage coefficient rho.
    The estimator is defined as the convex combination between the sample covariance and the target:
        Sigma_hat = (1 - rho) * Sample + rho * Target
    where rho is a shrinkage coefficient in [0, 1].

    The target matrix and the shrinkage coefficent can be set either at initialization or
    later using the corresponding properties. The rho parameter can also be passed to the fit method
    and will temporarily override the value set in the class. However, the object will continue storing the
    past value. If neither the fit method nor the class have a value for rho, an exception will be raised.
    """

    def __init__(
        self,
        target: np.ndarray | None = None,
        rho: float | None = None,
        stop_precision: bool = True,
        assume_centered: bool = True,
    ) -> None:
        super().__init__(stop_precision=stop_precision, assume_centered=assume_centered)

        self._target: np.ndarray | None = target
        self._rho: float | None = rho

    @property
    def target(self) -> np.ndarray:
        if self._target is None:
            raise TargetNotSetError()

        return self._target

    @target.setter
    def target(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise NotNumpyArrayError()
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise MatrixShapeComparisonError()
        self._target = value

    @property
    def rho(self) -> float:
        if self._rho is None:
            raise CoeffNotSetError()

        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        if not isinstance(value, (float, int)):
            raise TypeError("rho should be a float or an int.")

        if not (0 <= value <= 1):
            raise CoeffInvalidError("rho", value)

        self._rho = float(value)

    def _fit(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        The _fit method performs some validation checks on the provided arguments. At this level,
        although the number of arguments to be passed is not fixed, only the first positional two are considered.
        In particular, the first positional argument (if any) is considered as the shrinkage coefficient rho,
        while the second positional argument (if any) is ignored. The keyword argument 'rho' is also considered
        (if present) as the shrinkage coefficient. If both the positional and keyword arguments are provided,
        the positional argument takes precedence. If neither is provided, the method uses the value stored
        in the class instance. If no value is found, an exception is raised.
        After determining the value of rho, the method checks if the target matrix has been set. If not, an exception is raised.
        Finally, the method computes and returns the shrunk covariance matrix using the formula:
            Sigma_hat = (1 - rho) * Sample + rho * Target
        where Sample is the sample covariance matrix computed from the data X.

        The variable number of arguments is designed to be compatible with derived classes, like IdentityShrinkage,
        which may require additional parameters for their fitting process.
        """
        if not isinstance(X, np.ndarray):
            raise NotNumpyArrayError()

        if X.ndim != 2:
            raise MatrixShapeComparisonError()

        if args:
            rho = args[0]
        else:
            rho = kwargs.get("rho", None)

        rho_value = args[0] if args else kwargs.get("rho", None)
        if rho_value is None:
            if self._rho is None:
                raise CoeffNotSetError()
            rho = self._rho
        else:
            rho = float(rho_value)

        if self._target is None:
            raise TargetNotSetError()

        n_samples, _ = X.shape
        covariance: np.ndarray = (1 - rho) * np.dot(X.T, X) / n_samples + rho * self._target
        return covariance


class IdentityShrinkage(LinearShrinkage):
    """
    Linear Shrinkage estimator with identity target.
    The target matrix is set to mean(var) * I, where mean(var) is the mean of the variances
    of the features and I is the identity matrix. The shrinkage coefficient rho is estimated from the data
    using the Ledoit-Wolf approach, as described in the paper:

    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
    Journal of Multivariate Analysis, 88(2), 365-411.

    The user of this class can only set the stop_precision and assume_centered parameters at initialization.
    The target matrix and the shrinkage coefficient are automatically set during the fitting process.
    """

    def __init__(self, stop_precision: bool = True, assume_centered: bool = False) -> None:
        super().__init__(stop_precision=stop_precision, assume_centered=assume_centered)
        self._target: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape

        if not self._assume_centered:
            n_samples -= 1

        sample = np.dot(X.T, X) / n_samples

        diag = np.diag(sample)
        meanvar = sum(diag) / len(diag)
        self._target = meanvar * np.eye(n_features)

        Y2 = np.multiply(X, X)
        sample2 = np.dot(Y2.T, Y2) / n_samples
        piMat = sample2 - np.multiply(sample, sample)

        pi_hat = sum(piMat.sum(axis=1))

        gamma_hat = np.linalg.norm(sample - self._target, ord="fro") ** 2

        kappa_hat = pi_hat / gamma_hat
        rho = max(0, min(1, kappa_hat / n_samples))

        sigma_hat: np.ndarray = (1 - rho) * sample + rho * self._target
        self._rho = rho
        return sigma_hat
