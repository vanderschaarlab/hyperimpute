# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers
from hyperimpute.plugins.utils.simulate import simulate_nan


def dataset(mechanism: str, p_miss: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    n = 1000
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return x, x_miss, mask


def impute(
    plugin: ImputerPlugin, x: np.ndarray, x_miss: np.ndarray, mask: np.ndarray
) -> None:
    plugin.fit_transform(pd.DataFrame(x_miss))


@pytest.mark.slow
@pytest.mark.parametrize("plugin", ["sinkhorn"])
@pytest.mark.parametrize("mechanism", ["MAR", "MNAR", "MCAR"])
@pytest.mark.parametrize("p_miss", [0.1, 0.3, 0.5])
def test_perf(plugin: str, mechanism: str, p_miss: float, benchmark: Any) -> None:
    benchmark(impute, Imputers().get(plugin), *dataset(mechanism, p_miss))
