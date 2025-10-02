import numpy as np

np.random.seed(42)


def create_covariance(N: int) -> np.ndarray:
    """
    This function creates a random covariance matrix of size NxN,
    sampling its eigenvalues from a standard lognormal distribution.
    """

    if not isinstance(N, int | np.integer) or N <= 0:
        raise TypeError(f"N should be positive integers. Got {N} - {type(N)}.")

    lambdas = np.random.lognormal(mean=0, sigma=1, size=N)
    return np.diag(lambdas)


def validate_covariance(sigma: np.ndarray) -> bool:
    if not isinstance(sigma, np.ndarray):
        return False
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        print(sigma.shape)
        return False
    if not np.allclose(sigma, sigma.T):
        return False
    if np.any(np.linalg.eigvalsh(sigma) <= 0):
        return False
    return True


def create_data(N: int, p: int, sigma: np.ndarray) -> np.ndarray:
    """
    This function creates a random dataset of size NxP,
    sampling from a multivariate normal distribution with zero mean.
    If the covariance matrix is not provided,
    it will be created using the create_covariance function,
    which samples the eigenvalues from a lognormal distribution.
    """
    if not isinstance(N, int | np.integer) or N <= 0:
        raise TypeError(f"N should be positive integers. Got {N} - {type(N)}.")

    if not isinstance(p, int | np.integer) or p <= 0:
        raise TypeError(f"p should be positive integers. Got {p} - {type(p)}.")

    if sigma is None:
        Sigma = create_covariance(p)
    else:
        if not validate_covariance(sigma):
            raise ValueError("sigma should be a valid covariance matrix.")
        if sigma.shape[0] != p:
            raise ValueError(f"sigma should be of shape ({p}, {p}). Got {sigma.shape}.")

        Sigma = sigma
    return np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=N)


def PRIAL(sigma_hat: np.ndarray, sigma_star: np.ndarray, sigma: np.ndarray) -> float:
    """
    This function computes the Percentage Relative Improvement in Average Loss (PRIAL)
    of a covariance estimator with respect to the sample covariance matrix.
    sigma_hat: the sample covariance matrix.
    sigma_start: the covariance estimator.
    sigma: the true covariance matrix.
    The function returns the PRIAL value as a float.
    """

    if (
        not isinstance(sigma_hat, np.ndarray)
        or not isinstance(sigma_hat, np.ndarray)
        or not isinstance(sigma, np.ndarray)
    ):
        raise TypeError("All inputs should be numpy arrays.")

    if (
        sigma_hat.ndim != 2
        or sigma_hat.shape[0] != sigma_hat.shape[1]
        or sigma_hat.shape != sigma.shape
        or sigma.shape != sigma.shape
    ):
        raise ValueError("All inputs should be square matrices of the same size.")

    loss_sample = np.linalg.norm(sigma - sigma_hat, "fro") ** 2
    loss_estimator = np.linalg.norm(sigma - sigma_star, "fro") ** 2

    return float(100.0 * (loss_estimator - loss_sample) / loss_sample)
