# stdlib
from typing import Any

# third party
import numpy as np
import optuna
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# hyperimpute absolute
from hyperimpute.plugins.prediction import PredictionPlugin, Predictions
from hyperimpute.plugins.prediction.classifiers.plugin_neural_nets import plugin
from hyperimpute.utils.serialization import load, save
from hyperimpute.utils.tester import evaluate_estimator


def from_api() -> PredictionPlugin:
    return Predictions().get("neural_nets")


def from_module() -> PredictionPlugin:
    return plugin()


def from_pickle() -> PredictionPlugin:
    buff = save(plugin())
    return load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_pickle()])
def test_neural_nets_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_pickle()])
def test_neural_nets_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "neural_nets"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_pickle()])
def test_neural_nets_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "classifier"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_pickle()])
def test_neural_nets_plugin_hyperparams(test_plugin: PredictionPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 6
    assert test_plugin.hyperparameter_space()[0].name == "n_layers_hidden"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_pickle()])
def test_neural_nets_plugin_fit_predict(test_plugin: PredictionPlugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_pred = test_plugin.fit(X_train, y_train).predict(X_test)

    assert np.abs(np.subtract(y_pred.values, y_test.values)).mean() < 1


@pytest.mark.slow
def test_param_search() -> None:
    if len(plugin.hyperparameter_space()) == 0:
        return

    X, y = load_iris(return_X_y=True, as_frame=True)

    def evaluate_args(**kwargs: Any) -> float:
        kwargs["n_layers_hidden"] = 1
        kwargs["n_units_hidden"] = 2
        kwargs["n_iter"] = 10
        model = plugin(**kwargs)
        metrics = evaluate_estimator(model, X, y)

        return metrics["clf"]["aucroc"][0]

    def objective(trial: optuna.Trial) -> float:
        args = plugin.sample_hyperparameters(trial)
        return evaluate_args(**args)

    study = optuna.create_study(
        load_if_exists=True,
        directions=["maximize"],
        study_name=f"test_param_search_{plugin.name()}",
    )
    study.optimize(objective, n_trials=10, timeout=60)

    assert len(study.trials) == 10
