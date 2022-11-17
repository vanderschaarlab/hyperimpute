# stdlib
import multiprocessing
from typing import Any, List, Optional

# third party
from gpboost import GPBoostRegressor
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class GPBoostRegressorPlugin(base.RegressionPlugin):
    """Classification plugin based on the GPBoost classifier.

    Args:
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        max_depth: int
            Maximum depth of a tree.
        reg_lambda: float
            L2 regularization term on weights (xgb’s lambda).
        reg_alpha: float
            L1 regularization term on weights (xgb’s alpha).
        colsample_bytree: float
            Subsample ratio of columns when constructing each tree.
        subsample: float
            Subsample ratio of the training instance.
        learning_rate: float
            Boosting learning rate
        boosting_type: str
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
        random_state: float
            Random number seed.


    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("gpboost_regressor")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    boosting_type = ["gbdt", "goss", "dart"]

    def __init__(
        self,
        boosting_type: int = 0,
        max_depth: Optional[int] = -1,
        n_estimators: int = 100,
        reg_lambda: float = 0,
        reg_alpha: float = 0,
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        learning_rate: float = 1e-3,
        min_child_weight: int = 0.001,
        n_jobs: int = max(1, int(multiprocessing.cpu_count() / 2)),
        random_state: int = 0,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        self.model = GPBoostRegressor(
            boosting_type=GPBoostRegressorPlugin.boosting_type[boosting_type],
            n_estimators=n_estimators,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "gpboost_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("reg_lambda", 1e-3, 10.0),
            params.Float("reg_alpha", 1e-3, 10.0),
            params.Float("colsample_bytree", 0.1, 0.9),
            params.Float("subsample", 0.1, 0.9),
            params.Categorical("learning_rate", [1e-4, 1e-3, 1e-2]),
            params.Integer("max_depth", 2, 5),
            params.Integer("n_estimators", 10, 300),
            params.Integer("min_child_weight", 0, 300),
            params.Integer(
                "boosting_type", 0, len(GPBoostRegressorPlugin.boosting_type) - 1
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "GPBoostRegressorPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = GPBoostRegressorPlugin
