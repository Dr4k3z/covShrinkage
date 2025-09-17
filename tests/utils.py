import numpy as np
import pandas as pd


def create_df() -> tuple[np.ndarray, pd.DataFrame]:
    """
    This function creates a covariance matrix, picking values at random, and
    then generates a pd.DataFrame with a mutlivariate normal distribution
    with mean zero and the created covariance matrix.

    To ensure the randomly generated values actually do form a SPD matrix,
    we first create a (N x T) matrix of random standard normally distributed values and
    then multiply it by its transpose to get a (N x N) covariance matrix.
    """
    np.random.seed(42)

    N, T = 100, 150
    tmp = np.random.randn(N, T)
    sigma = tmp @ tmp.T

    df = pd.DataFrame(np.random.multivariate_normal(np.zeros(N), sigma, size=T))
    return sigma, df
