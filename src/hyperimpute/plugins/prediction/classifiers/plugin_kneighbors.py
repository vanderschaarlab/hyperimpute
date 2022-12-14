# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base


class KNeighborsClassifierPlugin(base.ClassifierPlugin):
    """Classification plugin based on the KNeighborsClassifier classifier.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("kneighbors")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    weights = ["uniform", "distance"]
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: int = 0,
        algorithm: int = 0,
        leaf_size: int = 30,
        p: int = 2,
        random_state: int = 0,
        hyperparam_search_iterations: Optional[int] = None,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if model is not None:
            self.model = model
            return

        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=KNeighborsClassifierPlugin.algorithm[algorithm],
            weights=KNeighborsClassifierPlugin.weights[weights],
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1,
        )

    @staticmethod
    def name() -> str:
        return "kneighbors"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer(
                "algorithm", 0, len(KNeighborsClassifierPlugin.algorithm) - 1
            ),
            params.Integer("weights", 0, len(KNeighborsClassifierPlugin.weights) - 1),
            params.Integer("n_neighbors", 5, 20),
            params.Integer("leaf_size", 5, 50),
            params.Integer("p", 1, 2),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "KNeighborsClassifierPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = KNeighborsClassifierPlugin
