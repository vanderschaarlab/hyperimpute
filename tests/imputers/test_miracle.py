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
from hyperimpute.plugins.imputers.plugin_miracle import plugin


def from_api() -> ImputerPlugin:
    return Imputers().get("miracle")


def from_module() -> ImputerPlugin:
    return plugin()


def from_serde() -> ImputerPlugin:
    buff = plugin().save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "miracle"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    res = test_plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    assert res.isnull().values.any() == False  # noqa


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
