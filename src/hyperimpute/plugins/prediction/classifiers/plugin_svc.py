# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.svm import SVC

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base


class SVCPlugin(base.ClassifierPlugin):
    """Regression plugin based on the SVM.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("svm")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    kernels = ["linear", "poly", "rbf", "sigmoid"]

    def __init__(
        self,
        kernel: int = 0,
        tol: float = 1e-3,
        C: float = 1.0,
        max_iter: int = -1,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)

        if hyperparam_search_iterations:
            max_iter = int(hyperparam_search_iterations) * 100

        self.model = SVC(
            kernel=SVCPlugin.kernels[kernel],
            tol=tol,
            C=C,
            max_iter=max_iter,
            probability=True,
        )

    @staticmethod
    def name() -> str:
        return "svc"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("C", 1e-3, 1),
            params.Float("tol", 1e-3, 1e-2),
            params.Integer("kernel", 0, len(SVCPlugin.kernels) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SVCPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = SVCPlugin
