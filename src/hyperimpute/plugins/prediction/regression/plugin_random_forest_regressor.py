# stdlib
import multiprocessing
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class RandomForestRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on Random forests.

    Method:
        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

    Args:
        n_estimators: int
            The number of trees in the forest.
        criterion: str
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        max_features: str
            The number of features to consider when looking for the best split.
        min_samples_split: int
            The minimum number of samples required to split an internal node.
        boostrap: bool
            Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        min_samples_leaf: int
            The minimum number of samples required to be at a leaf node.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("random_forest")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    criterions = ["squared_error", "absolute_error", "poisson"]
    features = ["auto", "sqrt", "log2"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: int = 0,
        max_features: int = 0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = 3,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=RandomForestRegressionPlugin.criterions[criterion],
            max_features=RandomForestRegressionPlugin.features[max_features],
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=max(1, int(multiprocessing.cpu_count() / 2)),
        )

    @staticmethod
    def name() -> str:
        return "random_forest_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer(
                "criterion", 0, len(RandomForestRegressionPlugin.criterions) - 1
            ),
            params.Integer(
                "max_features", 0, len(RandomForestRegressionPlugin.features) - 1
            ),
            params.Categorical("min_samples_split", [2, 5, 10]),
            params.Categorical("min_samples_leaf", [2, 5, 10]),
            params.Integer("max_depth", 1, 3),
            params.Integer("n_estimators", 10, 300, 10),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "RandomForestRegressionPlugin":
        if len(args) < 1:
            raise RuntimeError("please provide y for the fit method")

        X = np.asarray(X)
        y = np.asarray(args[0]).ravel()

        self.model.fit(X, y)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = RandomForestRegressionPlugin
