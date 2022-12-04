# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers
from hyperimpute.plugins.imputers.plugin_miwae import plugin
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.serialization import load, save


def from_serde() -> ImputerPlugin:
    buff = save(plugin(n_epochs=20))
    return load(buff)


def from_api() -> ImputerPlugin:
    return Imputers().get("miwae", n_epochs=20)


def from_module() -> ImputerPlugin:
    return plugin(n_epochs=20)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_miwae_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_miwae_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "miwae"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_miwae_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_miwae_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_miwae_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    res = test_plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    assert not np.all(np.isnan(res))


@pytest.mark.slow
@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
@pytest.mark.parametrize("mechanism", ["MAR"])
@pytest.mark.parametrize("p_miss", [0.3])
@pytest.mark.parametrize(
    "other_plugin",
    [Imputers().get("most_frequent")],
)
def test_compare_methods_perf(
    test_plugin: ImputerPlugin, mechanism: str, p_miss: float, other_plugin: Any
) -> None:
    np.random.seed(0)

    n = 10
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = pd.DataFrame(x_simulated["X_incomp"])

    x_ot = test_plugin.fit_transform(x_miss)
    rmse_ot = RMSE(x_ot.to_numpy(), x, mask)

    x_other = other_plugin.fit_transform(x_miss)
    rmse_other = RMSE(x_other.to_numpy(), x, mask)

    assert rmse_ot < rmse_other
