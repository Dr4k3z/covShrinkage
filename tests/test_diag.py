import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from covShrinkage.covDiag import covDiag

from covShrinkage.diag import DiagonalShrinkage

from .utils import read_matrix


@pytest.mark.parametrize(
    "data_file,reference_file",
    [
        ("X_200_4.txt", "2par_shrink_200_4.txt"),
        ("X_50_16.txt", "2par_shrink_50_16.txt"),
        ("X_16_50.txt", "2par_shrink_16_50.txt"),
    ],
)
def test_twoParams_shrinkage(data_file: str, reference_file: str) -> None:
    data_dir = Path(__file__).parent
    X = read_matrix(data_dir / "data" / data_file)

    raise NotImplementedError("DiagonalShrinkage not implemented yet.")

    S_hat = DiagonalShrinkage().fit(X).covariance
    S_hat_1 = covDiag(pd.DataFrame(X))
    true_S_hat = read_matrix(data_dir / "data" / reference_file)

    assert np.allclose(S_hat, S_hat_1)
    assert np.allclose(S_hat, true_S_hat)
    assert np.allclose(S_hat_1, true_S_hat)
