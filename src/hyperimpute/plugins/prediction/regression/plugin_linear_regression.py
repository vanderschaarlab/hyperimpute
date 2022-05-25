# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.linear_model import Ridge

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class LinearRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on the Linear Regression.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("linear_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

    def __init__(
        self,
        solver: int = 0,
        max_iter: Optional[int] = 10000,
        tol: float = 1e-3,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if hyperparam_search_iterations:
            max_iter = int(hyperparam_search_iterations) * 100

        self.model = Ridge(
            solver=LinearRegressionPlugin.solvers[solver],
            max_iter=max_iter,
        )

    @staticmethod
    def name() -> str:
        return "linear_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("max_iter", [100, 1000, 10000]),
            params.Integer("solver", 0, len(LinearRegressionPlugin.solvers) - 1),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "LinearRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = LinearRegressionPlugin
