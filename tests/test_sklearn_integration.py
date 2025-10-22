# type: ignore

import pickle
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from covShrinkage.linear import (
    ConstructorWarning,
    IdentityShrinkage,
    LinearShrinkage,
    TwoParametersShrinkage,
)

warnings.filterwarnings("ignore", category=ConstructorWarning)


@pytest.mark.parametrize(
    "estimator_cls",
    [IdentityShrinkage, LinearShrinkage, TwoParametersShrinkage],
)
def test_clone_linearshrinkage(estimator_cls: type[Any]) -> None:
    estimator = estimator_cls()
    auto = clone(estimator)
    assert isinstance(auto, LinearShrinkage)


@pytest.mark.parametrize(
    "estimator_cls",
    [IdentityShrinkage, LinearShrinkage, TwoParametersShrinkage],
)
def test_check_estimator_basic(estimator_cls: type[Any]) -> None:
    estimator = estimator_cls()
    check_estimator(estimator)


@parametrize_with_checks(
    [
        IdentityShrinkage(),
        LinearShrinkage(),
        TwoParametersShrinkage(),
    ]
)
def test_sklearn_compatibility(estimator: Any, check: Any) -> None:
    check(estimator)


@pytest.mark.parametrize(
    "estimator_cls",
    [IdentityShrinkage, LinearShrinkage, TwoParametersShrinkage],
)
def test_pipeline_usage(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    # Using StandardScaler then the estimator
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("cov", estimator_cls()),
        ]
    )
    pipe.fit(X)
    # If estimator has a .score method, call it
    score_val = pipe.named_steps["cov"].score(X)
    assert np.isfinite(score_val)


@pytest.mark.parametrize(
    "estimator_cls",
    [IdentityShrinkage, LinearShrinkage, TwoParametersShrinkage],
)
def test_grid_search_cv(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(1)
    X = rng.randn(50, 5)
    param_grid: dict[str, Any] = {
        "store_precision": [True, False],
        "assume_centered": [True, False],
    }
    gs = GridSearchCV(estimator_cls(), param_grid=param_grid, cv=3, scoring="r2")
    gs.fit(X)
    best = gs.best_estimator_
    assert isinstance(best, estimator_cls)
    assert best is not gs.estimator


@pytest.mark.parametrize(
    "estimator_cls",
    [IdentityShrinkage, LinearShrinkage, TwoParametersShrinkage],
)
def test_clone_and_pickle(estimator_cls: type[Any]) -> None:
    rng = np.random.RandomState(2)
    X = rng.randn(80, 8)
    est = estimator_cls(store_precision=True, assume_centered=False)
    est.fit(X)
    cov1 = est.covariance
    s = pickle.dumps(est)
    est2 = pickle.loads(s)
    cov2 = est2.covariance
    assert np.allclose(cov1, cov2)
