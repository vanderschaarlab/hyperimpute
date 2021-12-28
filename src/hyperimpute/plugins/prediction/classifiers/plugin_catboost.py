# stdlib
from typing import Any, List, Optional

# third party
from catboost import CatBoostClassifier
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base
import hyperimpute.utils.serialization as serialization


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

    grow_policies = ["Depthwise", "SymmetricTree", "Lossguide"]

    def __init__(
        self,
        n_estimators: int = 100,
        depth: int = 5,
        grow_policy: int = 0,
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

        self.model = CatBoostClassifier(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="6gb",
            n_estimators=n_estimators,
            grow_policy=CatBoostPlugin.grow_policies[grow_policy],
        )

    @staticmethod
    def name() -> str:
        return "catboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("depth", 4, 7),
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

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CatBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = CatBoostPlugin
