# stdlib
from typing import Any, List

# third party
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
from hyperimpute.plugins.imputers.plugin_hyperimpute import plugin as base_model
import hyperimpute.plugins.utils.decorators as decorators


class MissForestPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the MissForest strategy.

    Method:
        Iterative chained equations(ICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a ExtraTreesRegressor, which fits a number of randomized extra-trees and averages the results.

    Paper: "MissForest—non-parametric missing value imputation for mixed-type data", Daniel J. Stekhoven, Peter Bühlmann


    Args:
        n_estimators: int, default=10
            The number of trees in the forest.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    HyperImpute Hyperparameters:
        n_estimators: The number of trees in the forest.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("missforest")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_iter: int = 100,
        initial_strategy: int = 0,
        imputation_order: int = 0,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order

        self._model = base_model(
            classifier_seed=["random_forest"],
            regression_seed=["random_forest_regressor"],
            imputation_order=imputation_order,
            baseline_imputer=initial_strategy,
            random_state=random_state,
            n_inner_iter=max_iter,
            class_threshold=5,
        )

    @staticmethod
    def name() -> str:
        return "missforest"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 10, 50, 10),
            params.Integer("max_iter", 100, 300, 100),
            params.Integer("max_depth", 1, 3),
        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MissForestPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)


plugin = MissForestPlugin
