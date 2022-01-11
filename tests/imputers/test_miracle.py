# third party
import numpy as np
import pandas as pd
import pytest

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

    assert res.isnull().values.any() == False
