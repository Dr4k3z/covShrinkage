import numpy as np

from covShrinkage.cov1Para import cov1Para

from .utils import create_df


def test_covPara1() -> None:
    true_sigma, df = create_df()

    sigmahat = cov1Para(df)
    sample_cov = np.cov(df, rowvar=False)

    sample_err = np.linalg.norm(true_sigma - sample_cov, ord="fro") / np.linalg.norm(
        true_sigma, ord="fro"
    )
    shrink_err = np.linalg.norm(true_sigma - sigmahat, ord="fro") / np.linalg.norm(
        true_sigma, ord="fro"
    )

    assert shrink_err < sample_err
