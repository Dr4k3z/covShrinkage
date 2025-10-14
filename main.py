import numpy as np
import pandas as pd

from covShrinkage.linear import LinearShrinkage

N, T = 100, 150


def create_df() -> tuple[np.ndarray, pd.DataFrame]:
    np.random.seed(42)

    tmp = np.random.randn(N, T)
    sigma = tmp @ tmp.T

    df = pd.DataFrame(np.random.multivariate_normal(np.zeros(N), sigma, size=T))
    return sigma, df


if __name__ == "__main__":
    sigma, df = create_df()
    sample_cov = np.cov(df, rowvar=False)

    sigmahat = LinearShrinkage(assume_centered=True)
    sigmahat.target = np.eye(N)
    sigmahat.fit(df.values, rho=0.1)

    print(
        f"Error: {np.linalg.norm(sigma - sigmahat, ord='fro') / np.linalg.norm(sigma, ord='fro')}"
    )
    print(
        f"Sample Cov Error: {np.linalg.norm(sigma - sample_cov, ord='fro') / np.linalg.norm(sigma, ord='fro')}"
    )
