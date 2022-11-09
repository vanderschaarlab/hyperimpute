# stdlib
from typing import Any, List, Optional

# third party
from catboost import CatBoostRegressor
import pandas as pd

# hyperimpute absolute
from hyperimpute.plugins.core.device import DEVICE
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base


class CatBoostRegressorPlugin(base.RegressionPlugin):
    """Regression plugin based on the CatBoost framework.

    Method:
        CatBoost provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. It uses Ordered Boosting to overcome over fitting and Symmetric Trees for faster execution.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("catboost_regressor")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    grow_policies: List[Optional[str]] = [
        None,
        "Depthwise",
        "SymmetricTree",
        "Lossguide",
    ]

    def __init__(
        self,
        depth: Optional[int] = None,
        grow_policy: int = 0,
        n_estimators: Optional[int] = 10,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        l2_leaf_reg: float = 3,
        learning_rate: float = 1e-3,
        min_data_in_leaf: int = 1,
        random_strength: float = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        gpu_args = {}

        if DEVICE == "cuda":
            gpu_args = {
                "task_type": "GPU",
            }

        self.model = CatBoostRegressor(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="20gb",
            n_estimators=n_estimators,
            grow_policy=CatBoostRegressorPlugin.grow_policies[grow_policy],
            random_state=random_state,
            l2_leaf_reg=l2_leaf_reg,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            random_strength=random_strength,
            **gpu_args,
        )

    @staticmethod
    def name() -> str:
        return "catboost_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("depth", 1, 5),
            params.Integer("n_estimators", 10, 100),
            params.Integer(
                "grow_policy", 0, len(CatBoostRegressorPlugin.grow_policies) - 1
            ),
            params.Float("learning_rate", 1e-2, 4e-2),
            params.Float("l2_leaf_reg", 1e-4, 1e3),
            params.Float("random_strength", 0, 3),
            params.Integer("min_data_in_leaf", 1, 300),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CatBoostRegressorPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = CatBoostRegressorPlugin
