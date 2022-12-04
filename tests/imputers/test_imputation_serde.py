# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins import Imputers
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.serialization import load, save


def dataset(mechanism: str, p_miss: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    n = 20
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    x_miss = x_simulated["X_incomp"]

    return pd.DataFrame(x), pd.DataFrame(x_miss)


@pytest.mark.slow
@pytest.mark.parametrize("plugin", Imputers().list())
def test_pickle(plugin: str) -> None:
    x, x_miss = dataset("MAR", 0.3)

    estimator = Imputers().get(plugin)

    estimator.fit_transform(x_miss)

    buff = save(estimator)
    estimator_new = load(buff)

    estimator_new.transform(x_miss)
