# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from xgboost import XGBRegressor

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class XGBoostRegressorPlugin(base.RegressionPlugin):
    """Classification plugin based on the XGBoostRegressor.

    Method:
        Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoostRegressor algorithm has a robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune.

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
        colsample_bynode: float
             Subsample ratio of columns for each split.
        colsample_bylevel: float
             Subsample ratio of columns for each level.
        subsample: float
            Subsample ratio of the training instance.
        learning_rate: float
            Boosting learning rate
        booster: str
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
        max_bin: int
            Number of bins for histogram construction.
        tree_method: str
            Specify which tree method to use. Default to auto. If this parameter is set to default, XGBoostRegressor will choose the most conservative option available.
        random_state: float
            Random number seed.


    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regressors").get("xgboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        lr: float = 0.01,
        random_state: int = 0,
        model: Any = None,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        self.model = XGBRegressor(
            verbosity=0,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            nthread=-1,
            lr=lr,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "xgboost_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_depth", 2, 9),
            params.Categorical("lr", [0.01, 0.1]),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "XGBoostRegressorPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = XGBoostRegressorPlugin
