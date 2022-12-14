# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.linear_model import LogisticRegression

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.prediction.classifiers.base as base


class LogisticRegressionPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Logistic Regression classifier.

    Method:
        Logistic regression is a linear model for classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

    Args:
        C: float
            Inverse of regularization strength; must be a positive float.
        solver: str
            Algorithm to use in the optimization problem: [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]
        multi_class: str
            If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
        class_weight: str
            Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one.
        max_iter: int
            Maximum number of iterations taken for the solvers to converge.

    Example:
        >>> from hyperimpute.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("logistic_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    solvers = ["newton-cg", "lbfgs", "sag", "saga"]
    classes = ["auto", "ovr", "multinomial"]
    weights = ["balanced", None]

    def __init__(
        self,
        C: float = 1.0,
        solver: int = 1,
        multi_class: int = 0,
        class_weight: int = 0,
        max_iter: int = 10000,
        penalty: str = "l2",
        model: Any = None,
        random_state: int = 0,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        if model is not None:
            self.model = model
            return

        self.model = LogisticRegression(
            C=C,
            solver=LogisticRegressionPlugin.solvers[solver],
            multi_class=LogisticRegressionPlugin.classes[multi_class],
            class_weight=LogisticRegressionPlugin.weights[class_weight],
            penalty=penalty,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
        )

    @staticmethod
    def name() -> str:
        return "logistic_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("C", 1e-3, 1e-2),
            params.Integer("solver", 0, len(LogisticRegressionPlugin.solvers) - 1),
            params.Integer("multi_class", 0, len(LogisticRegressionPlugin.classes) - 1),
            params.Integer(
                "class_weight", 0, len(LogisticRegressionPlugin.weights) - 1
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "LogisticRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)


plugin = LogisticRegressionPlugin
