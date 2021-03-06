# stdlib
from typing import Any, List, Optional

# third party
from catboost import CatBoostClassifier
import pandas as pd

# hyperimpute absolute
from hyperimpute.plugins.core.device import DEVICE
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base


class CatBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the CatBoost framework.

    Method:
        CatBoost provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. It uses Ordered Boosting to overcome over fitting and Symmetric Trees for faster execution.

    Args:
        learning_rate: float
            The learning rate used for training.
        depth: int

        iterations: int
        grow_policy: int

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("catboost")
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
        n_estimators: Optional[int] = 10,
        depth: Optional[int] = None,
        grow_policy: int = 0,
        model: Any = None,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        gpu_args = {}

        if DEVICE == "cuda":
            gpu_args = {
                "task_type": "GPU",
            }

        self.model = CatBoostClassifier(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="6gb",
            n_estimators=n_estimators,
            grow_policy=CatBoostPlugin.grow_policies[grow_policy],
            random_state=random_state,
            **gpu_args,
        )

    @staticmethod
    def name() -> str:
        return "catboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("depth", 1, 5),
            params.Integer("n_estimators", 10, 100),
            params.Integer("grow_policy", 0, len(CatBoostPlugin.grow_policies) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CatBoostPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = CatBoostPlugin
