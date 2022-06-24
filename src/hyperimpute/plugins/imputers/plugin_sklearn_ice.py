# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base


class SKLearnIterativeChainedEquationsPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Multivariate Iterative chained equations Imputation strategy.

    Method:
        Multivariate Iterative chained equations(MICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a BayesianRidge estimator, which does a regularized linear regression.

    Args:
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("ice")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    """

    initial_strategy_vals = ["mean", "median", "most_frequent", "constant"]
    imputation_order_vals = ["ascending", "descending", "roman", "arabic", "random"]

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 0.001,
        initial_strategy: int = 0,
        imputation_order: int = 0,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order

        self._model = IterativeImputer(
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            initial_strategy=SKLearnIterativeChainedEquationsPlugin.initial_strategy_vals[
                initial_strategy
            ],
            imputation_order=SKLearnIterativeChainedEquationsPlugin.imputation_order_vals[
                imputation_order
            ],
            sample_posterior=False,
        )

    @staticmethod
    def name() -> str:
        return "sklearn_ice"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_iter", 100, 1000, 100),
            params.Categorical("tol", [1e-2, 1e-3, 1e-4]),
            params.Integer(
                "initial_strategy",
                0,
                len(SKLearnIterativeChainedEquationsPlugin.initial_strategy_vals) - 1,
            ),
            params.Integer(
                "imputation_order",
                0,
                len(SKLearnIterativeChainedEquationsPlugin.imputation_order_vals) - 1,
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "SKLearnIterativeChainedEquationsPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)


plugin = SKLearnIterativeChainedEquationsPlugin
