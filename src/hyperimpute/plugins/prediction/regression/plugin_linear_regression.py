# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import LinearRegression

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base
import hyperimpute.utils.serialization as serialization


class LinearRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on the Linear Regression classifier.

    Method:
        Linear regression is a linear model for classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a linear function.

    Args:
        C: float
            Inverse of regularization strength; must be a positive float.
        solver: str
            Algorithm to use in the optimization problem: [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
        multi_class: str
            If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
        class_weight: str
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        max_iter: int
            Maximum number of iterations taken for the solvers to converge.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("linear_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(self, model: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.model = LinearRegression(
            n_jobs=-1,
        )

    @staticmethod
    def name() -> str:
        return "linear_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "LinearRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "LinearRegressionPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = LinearRegressionPlugin
