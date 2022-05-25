# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.neural_network import MLPRegressor

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class MLPRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on the MLP Regression classifier.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("mlp_regressor")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(self, model: Any = None, random_state: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.model = MLPRegressor(max_iter=500, random_state=random_state)

    @staticmethod
    def name() -> str:
        return "mlp_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MLPRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = MLPRegressionPlugin
