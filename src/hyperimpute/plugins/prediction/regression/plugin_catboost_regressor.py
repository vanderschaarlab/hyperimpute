# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# hyperimpute absolute
from hyperimpute.plugins.core.device import DEVICE
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.regression.base as base
from hyperimpute.utils.pip import install

for retry in range(2):
    try:
        # Necessary packages
        # third party
        from catboost import CatBoostRegressor

        break
    except ImportError:
        depends = ["catboost"]
        install(depends)


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
            used_ram_limit="6gb",
            n_estimators=n_estimators,
            grow_policy=CatBoostRegressorPlugin.grow_policies[grow_policy],
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
