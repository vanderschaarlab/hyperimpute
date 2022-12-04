# third party
import numpy as np
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers
from hyperimpute.plugins.imputers.plugin_mean import plugin
from hyperimpute.utils.serialization import load, save


def from_serde() -> ImputerPlugin:
    buff = save(plugin())
    return load(buff)


def from_api() -> ImputerPlugin:
    return Imputers().get("mean")


def from_module() -> ImputerPlugin:
    return plugin()


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "mean"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_mean_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    res = test_plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 4, 4], [3, 3, 9, 9], [2, 2, 2, 2]]
    )
