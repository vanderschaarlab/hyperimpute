# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers
from hyperimpute.plugins.imputers.plugin_hyperimpute import plugin
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.serialization import load, save


def from_serde(optimizer: str = "simple") -> ImputerPlugin:
    buff = save(
        plugin(
            classifier_seed=["logistic_regression", "random_forests"],
            regression_seed=["linear_regression", "random_forests_regressor"],
            optimizer=optimizer,
        )
    )
    return load(buff)


def from_api(
    optimizer: str = "simple",
    classifier_seed: list = ["logistic_regression", "random_forests"],
    regression_seed: list = ["linear_regression", "random_forests_regressor"],
    imputation_order: int = 0,
) -> ImputerPlugin:
    return Imputers().get(
        "hyperimpute",
        classifier_seed=classifier_seed,
        regression_seed=regression_seed,
        optimizer=optimizer,
        imputation_order=imputation_order,
    )


def from_module(optimizer: str = "simple") -> ImputerPlugin:
    return plugin(
        optimizer=optimizer,
        classifier_seed=["logistic_regression", "random_forests"],
        regression_seed=["linear_regression", "random_forests_regressor"],
    )


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_hyperimpute_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_hyperimpute_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "hyperimpute"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_hyperimpute_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_hyperimpute_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_hyperimpute_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    plugin = test_plugin
    res = plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )
    assert not np.any(np.isnan(res))


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
@pytest.mark.parametrize("mechanism", ["MAR"])
@pytest.mark.parametrize("p_miss", [0.5])
@pytest.mark.parametrize(
    "other_plugin",
    [Imputers().get("most_frequent")],
)
def test_compare_methods_perf(
    test_plugin: ImputerPlugin, mechanism: str, p_miss: float, other_plugin: Any
) -> None:
    np.random.seed(0)

    n = 20
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = pd.DataFrame(x_simulated["X_incomp"])

    x_mf = test_plugin.fit_transform(x_miss)
    rmse_mf = RMSE(x_mf.to_numpy(), x, mask)

    x_other = other_plugin.fit_transform(x_miss)
    rmse_other = RMSE(x_other.to_numpy(), x, mask)

    assert rmse_mf < rmse_other


@pytest.mark.parametrize("optimizer", ["hyperband", "bayesian", "simple"])
@pytest.mark.parametrize("mechanism", ["MAR"])
@pytest.mark.parametrize("p_miss", [0.5])
@pytest.mark.parametrize(
    "other_plugin",
    [Imputers().get("most_frequent")],
)
def test_compare_optimizers(
    optimizer: str, mechanism: str, p_miss: float, other_plugin: Any
) -> None:
    test_plugin = from_api(optimizer)

    np.random.seed(0)

    n = 20
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = pd.DataFrame(x_simulated["X_incomp"])

    x_mf = test_plugin.fit_transform(x_miss)
    rmse_mf = RMSE(x_mf.to_numpy(), x, mask)

    x_other = other_plugin.fit_transform(x_miss)
    rmse_other = RMSE(x_other.to_numpy(), x, mask)

    assert rmse_mf < rmse_other


@pytest.mark.parametrize("imputation_order", [0, 1, 2])
@pytest.mark.parametrize("mechanism", ["MAR"])
@pytest.mark.parametrize("p_miss", [0.5])
@pytest.mark.parametrize(
    "other_plugin",
    [Imputers().get("most_frequent")],
)
def test_imputation_order(
    imputation_order: int, mechanism: str, p_miss: float, other_plugin: Any
) -> None:
    test_plugin = from_api(
        imputation_order=imputation_order,
        classifier_seed=["logistic_regression"],
        regression_seed=["linear_regression"],
    )

    np.random.seed(0)

    n = 20
    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = pd.DataFrame(x_simulated["X_incomp"])

    x_mf = test_plugin.fit_transform(x_miss)
    rmse_mf = RMSE(x_mf.to_numpy(), x, mask)

    x_other = other_plugin.fit_transform(x_miss)
    rmse_other = RMSE(x_other.to_numpy(), x, mask)

    assert rmse_mf < rmse_other
