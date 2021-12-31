# stdlib
import copy
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

# hyperimpute absolute
import hyperimpute.logger as log
from hyperimpute.utils.metrics import evaluate_auc, generate_score, print_score


class Eval:
    """Helper class for evaluating the performance of the models.

    Args:
        metric: str, default="aucroc"
            The type of metric to use for evaluation. Potential values: ["aucprc", "aucroc"].
    """

    def __init__(self, metric: str = "aucroc") -> None:
        metric_allowed = ["aucprc", "aucroc"]

        if metric not in metric_allowed:
            raise ValueError(
                f"invalid metric {metric}. supported values are {metric_allowed}"
            )
        self.m_metric = metric

    def get_metric(self) -> str:
        return self.m_metric

    def score_proba(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:
        if y_test is None:
            raise RuntimeError("invalid y_test")
        if y_pred_proba is None:
            raise RuntimeError("Invalid y_pred_proba")

        if self.m_metric == "aucprc":
            score_val = self.average_precision_score(y_test, y_pred_proba)
        elif self.m_metric == "aucroc":
            score_val = self.roc_auc_score(y_test, y_pred_proba)
        else:
            raise ValueError(f"invalid metric {self.m_metric}")

        log.debug(f"evaluate:{score_val:0.5f}")
        return score_val

    def roc_auc_score(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:

        return evaluate_auc(y_test, y_pred_proba)[0]

    def average_precision_score(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:

        return evaluate_auc(y_test, y_pred_proba)[1]


def evaluate_estimator(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    metric: str = "aucroc",
    seed: int = 0,
    pretrained: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metric_ = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    ev = Eval(metric)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)

        metric_[indx] = ev.score_proba(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


def score_classification_model(
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> float:
    model = copy.deepcopy(estimator)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)


def evaluate_regression(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    metric: str = "rmse",
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metric_ = np.zeros(n_folds)

    indx = 0
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = mean_squared_error(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }
