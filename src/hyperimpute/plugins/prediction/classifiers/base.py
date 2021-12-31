# stdlib
from typing import Any

# third party
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.base_plugin as plugin
import hyperimpute.plugins.prediction.base as prediction_base
import hyperimpute.plugins.utils.cast as cast
from hyperimpute.utils.tester import Eval


class ClassifierPlugin(prediction_base.PredictionPlugin):
    """Base class for the classifier plugins.

    It provides the implementation for plugin.Plugin's subtype, _fit and _predict methods.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.args = kwargs

        super().__init__()

    @staticmethod
    def subtype() -> str:
        return "classifier"

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> plugin.Plugin:
        X = cast.to_dataframe(X)

        if len(args) == 0:
            raise RuntimeError("Please provide the training labels as well")

        Y = cast.to_dataframe(args[0]).values.ravel()

        return self._fit(X, Y, **kwargs)

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "aucroc") -> float:
        ev = Eval(metric)

        preds = self.predict_proba(X)
        return ev.score_proba(y, preds)

    def get_args(self) -> dict:
        return self.args
