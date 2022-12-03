# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# hyperimpute absolute
from hyperimpute.plugins.imputers import Imputers


def _eval_imputer_pipeline(test_imputer: str) -> None:
    if test_imputer == "nop" or test_imputer == "EM":
        return

    imputer = Imputers().get(test_imputer)

    rng = np.random.RandomState(0)

    X_full, y_full = load_iris(as_frame=True, return_X_y=True)

    n_samples = int(X_full.shape[0])
    n_features = int(X_full.shape[1])

    # Add missing values in 70% of the lines
    missing_rate = 0.7
    n_missing_samples = int(np.floor(n_samples * missing_rate))
    missing_samples = np.hstack(
        (
            np.zeros(n_samples - n_missing_samples, dtype=np.bool),
            np.ones(n_missing_samples, dtype=np.bool),
        )
    )
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)

    # Estimate the score without the lines containing missing values
    X_filtered = X_full.values[~missing_samples, :]
    y_filtered = y_full.values[~missing_samples]
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    score_with_miss = cross_val_score(estimator, X_filtered, y_filtered, cv=2).mean()
    print(f"Score without the samples containing missing values = {score_with_miss}")

    # Estimate the score after imputation of the missing values
    X_missing = X_full.copy().values
    X_missing[np.where(missing_samples)[0], missing_features] = np.nan
    X_missing = pd.DataFrame(X_missing, index=X_full.index)

    y_missing = y_full.copy()
    estimator = Pipeline(
        [
            ("imputer", imputer),
            ("forest", RandomForestRegressor(random_state=0, n_estimators=100)),
        ]
    )
    score_with_impute = cross_val_score(
        estimator, X_missing, y_missing, cv=2, error_score="raise"
    ).mean()
    print(
        f"Score after imputation of the missing values using {imputer.name()} = {score_with_impute}"
    )

    assert score_with_miss < score_with_impute


@pytest.mark.slow
@pytest.mark.parametrize("test_imputer", Imputers().list())
def test_sklearn_imputation_pipeline_full(test_imputer: str) -> None:
    return _eval_imputer_pipeline(test_imputer)
