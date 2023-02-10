# stdlib
from typing import Any

# third party
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import load_iris

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers
from hyperimpute.plugins.imputers.plugin_sinkhorn import plugin
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.serialization import load, save


def from_serde() -> ImputerPlugin:
    buff = save(plugin(n_epochs=20))
    return load(buff)


def from_api() -> ImputerPlugin:
    return Imputers().get("sinkhorn", n_epochs=20)


def from_module() -> ImputerPlugin:
    return plugin(n_epochs=20)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_sinkhorn_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_sinkhorn_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "sinkhorn"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_sinkhorn_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_sinkhorn_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 6


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_sinkhorn_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    res = test_plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    assert not np.all(np.isnan(res))


@pytest.mark.slow
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

    n = 50
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


def test_param_search() -> None:
    if len(plugin.hyperparameter_space()) == 0:
        return

    X, _ = load_iris(return_X_y=True)
    orig_val = X[0, 0]
    X[0, 0] = np.nan

    def evaluate_args(**kwargs: Any) -> float:
        X_imp = plugin(**kwargs).fit_transform(X.copy()).values

        return np.abs(orig_val - X_imp[0, 0])

    def objective(trial: optuna.Trial) -> float:
        args = plugin.sample_hyperparameters(trial)
        return evaluate_args(**args)

    study = optuna.create_study(
        load_if_exists=True,
        directions=["minimize"],
        study_name=f"test_param_search_{plugin.name()}",
    )
    study.optimize(objective, n_trials=10, timeout=60)

    assert len(study.trials) > 0
