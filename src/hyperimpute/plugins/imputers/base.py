# stdlib
from typing import Any

# third party
import pandas as pd
from sklearn.impute._base import _BaseImputer

# hyperimpute absolute
import hyperimpute.plugins.core.base_plugin as plugin
from hyperimpute.utils.distributions import enable_reproducible_results


class ImputerPlugin(_BaseImputer, plugin.Plugin):
    """Base class for the imputation plugins.

    It provides the implementation for plugin.Plugin.type() static method.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin. e.g., EM, mice, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.
        _fit() - internal implementation, called by the `fit()` method.
        _transform() - internal implementation, called by the `transform()` method.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self, random_state: int = 0) -> None:
        _BaseImputer.__init__(self)
        plugin.Plugin.__init__(self)

        enable_reproducible_results(random_state)
        self.random_state = random_state

    @staticmethod
    def type() -> str:
        return "imputer"

    @staticmethod
    def subtype() -> str:
        return "default"

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "Imputation plugins do not implement the 'predict' method"
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Imputation plugins do not implement the 'predict_proba' method"
        )
