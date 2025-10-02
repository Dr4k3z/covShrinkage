"""
This unit test file tests the IdentityShrinkage class in the linear module.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from covShrinkage.cov1Para import cov1Para
from covShrinkage.linear import IdentityShrinkage

from .utils import read_matrix


def test_identity_shrinkage_200_4() -> None:
    data_dir = Path(__file__).parent
    filename = data_dir / "data/X_200_4.txt"
    X = read_matrix(filename)

    S_hat = IdentityShrinkage().fit(X).covariance
    S_hat_1 = cov1Para(pd.DataFrame(X))
    true_S_hat = read_matrix(data_dir / "data/id_shrink_200_4.txt")

    for i in range(S_hat.shape[0]):
        for j in range(S_hat.shape[1]):
            print(f"S_hat[{i}, {j}] = {S_hat[i, j]}, true_S_hat[{i}, {j}] = {true_S_hat[i, j]}")

    assert np.allclose(S_hat, S_hat_1)
    assert np.allclose(S_hat, true_S_hat)
    assert np.allclose(S_hat_1, true_S_hat)


def test_identity_shrinkage_50_16() -> None:
    data_dir = Path(__file__).parent
    filename = data_dir / "data/X_50_16.txt"
    X = read_matrix(filename)

    S_hat = IdentityShrinkage().fit(X).covariance
    S_hat_1 = cov1Para(pd.DataFrame(X))
    true_S_hat = read_matrix(data_dir / "data/id_shrink_50_16.txt")

    for i in range(S_hat.shape[0]):
        for j in range(S_hat.shape[1]):
            print(f"S_hat[{i}, {j}] = {S_hat[i, j]}, true_S_hat[{i}, {j}] = {true_S_hat[i, j]}")

    assert np.allclose(S_hat, S_hat_1)
    assert np.allclose(S_hat, true_S_hat)
    assert np.allclose(S_hat_1, true_S_hat)


def test_identity_shrinkage_16_50() -> None:
    data_dir = Path(__file__).parent
    filename = data_dir / "data/X_16_50.txt"
    X = read_matrix(filename)

    S_hat = IdentityShrinkage().fit(X).covariance
    S_hat_1 = cov1Para(pd.DataFrame(X))
    true_S_hat = read_matrix(data_dir / "data/id_shrink_16_50.txt")

    for i in range(S_hat.shape[0]):
        for j in range(S_hat.shape[1]):
            print(f"S_hat[{i}, {j}] = {S_hat[i, j]}, true_S_hat[{i}, {j}] = {true_S_hat[i, j]}")

    assert np.allclose(S_hat, S_hat_1)
    assert np.allclose(S_hat, true_S_hat)
    assert np.allclose(S_hat_1, true_S_hat)
