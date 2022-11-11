# stdlib
from typing import Any, List

# third party
import lightgbm as lgbm
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class LGBMRegressorPlugin(base.RegressionPlugin):
    """Regression plugin based on LGBMRegressor.

    Method:
        Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient boosted trees, which usually outperforms random forest.

    Args:
        n_estimators: int
            The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        learning_rate: float
            Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
        max_depth: int
            The maximum depth of the individual regression estimators.
        boosting_type: str
            ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
        objective:str
             Specify the learning task and the corresponding learning objective or a custom objective function to be used.
        reg_lambda:float
             L2 regularization term on weights.
        reg_alpha:float
             L1 regularization term on weights.
        colsample_bytree:float
            Subsample ratio of columns when constructing each tree.
        subsample:float
            Subsample ratio of the training instance.
        num_leaves:int
             Maximum tree leaves for base learners.
        min_child_samples:int
            Minimum sum of instance weight (hessian) needed in a child (leaf).

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("lgbm")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        n_estimators: int = 100,
        boosting_type: str = "gbdt",
        learning_rate: float = 1e-2,
        max_depth: int = 6,
        reg_lambda: float = 1e-3,
        reg_alpha: float = 1e-3,
        colsample_bytree: float = 0.1,
        subsample: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 1,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.model = lgbm.LGBMRegressor(
            n_estimators=n_estimators,
            boosting_type=boosting_type,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
        )

    @staticmethod
    def name() -> str:
        return "lgbm_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 5, 100),
            params.Float("reg_lambda", 1e-3, 1e3),
            params.Float("reg_alpha", 1e-3, 1e3),
            params.Float("colsample_bytree", 0.1, 1.0),
            params.Float("subsample", 0.1, 1.0),
            params.Integer("num_leaves", 31, 256),
            params.Integer("min_child_samples", 1, 500),
            params.Categorical("learning_rate", [1e-4, 1e-3, 1e-2, 2e-4]),
            params.Integer("max_depth", 1, 6),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "LGBMRegressorPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)


plugin = LGBMRegressorPlugin
