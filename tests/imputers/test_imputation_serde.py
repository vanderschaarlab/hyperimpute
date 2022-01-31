# stdlib
from typing import Tuple

# third party
import numpy as np
import pytest

# hyperimpute absolute
from hyperimpute.plugins import Imputers
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.serialization import load_model, save_model


def dataset(mechanism: str, p_miss: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    n = 20
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return x, x_miss, mask


@pytest.mark.slow
@pytest.mark.parametrize("plugin", Imputers().list_available())
def test_pickle(plugin: str) -> None:
    x, x_miss, mask = dataset("MAR", 0.3)

    estimator = Imputers().get(plugin)

    estimator.fit_transform(x_miss)

    buff = save_model(estimator)
    estimator_new = load_model(buff)

    estimator_new.transform(x_miss)
