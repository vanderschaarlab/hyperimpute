# stdlib
import multiprocessing
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base


class XGBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the XGBoost classifier.

    Method:
        Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm has a robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune.

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
        lr: float
            Boosting learning rate
        booster: str
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
        max_bin: int
            Number of bins for histogram construction.
        random_state: float
            Random number seed.


    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("xgboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    booster = ["gbtree", "gblinear", "dart"]
    grow_policy = ["depthwise", "lossguide"]

    def __init__(
        self,
        n_estimators: int = 100,
        reg_lambda: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        max_depth: Optional[int] = 3,
        subsample: Optional[float] = None,
        lr: Optional[float] = None,
        min_child_weight: Optional[int] = None,
        max_bin: int = 256,
        booster: int = 0,
        grow_policy: int = 0,
        nthread: int = max(1, int(multiprocessing.cpu_count() / 2)),
        random_state: int = 0,
        eta: float = 0.3,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        gpu_args = {}

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            max_depth=max_depth,
            subsample=subsample,
            lr=lr,
            min_child_weight=min_child_weight,
            max_bin=max_bin,
            eta=eta,
            verbosity=0,
            use_label_encoder=False,
            grow_policy=XGBoostPlugin.grow_policy[grow_policy],
            # booster=XGBoostPlugin.booster[booster],
            random_state=random_state,
            nthread=nthread,
            **gpu_args,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "xgboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("eta", 1e-3, 0.5),
            params.Float("reg_lambda", 1e-3, 10.0),
            params.Float("reg_alpha", 1e-3, 10.0),
            params.Float("colsample_bytree", 0.1, 0.9),
            params.Float("colsample_bynode", 0.1, 0.9),
            params.Float("colsample_bylevel", 0.1, 0.9),
            params.Float("subsample", 0.1, 0.9),
            params.Categorical("lr", [1e-4, 1e-3, 1e-2]),
            params.Integer("max_depth", 2, 5),
            params.Integer("n_estimators", 10, 300),
            params.Integer("min_child_weight", 0, 300),
            params.Integer("max_bin", 256, 512),
            params.Integer("grow_policy", 0, len(XGBoostPlugin.grow_policy) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "XGBoostPlugin":
        y = np.asarray(args[0])
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)
        self.model.fit(X, y, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.encoder.inverse_transform(self.model.predict(X, *args, **kwargs))

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = XGBoostPlugin
