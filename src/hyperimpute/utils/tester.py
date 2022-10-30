# stdlib
import copy
from typing import Any, Dict, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# hyperimpute absolute
import hyperimpute.logger as log
from hyperimpute.utils.metrics import (
    evaluate_auc,
    evaluate_wnd,
    generate_score,
    print_score,
)


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
        if self.m_metric != "aucroc":
            raise RuntimeError("Invalid metric for the evaluator. expected AUCROC")

        return evaluate_auc(y_test, y_pred_proba, self.m_metric)

    def average_precision_score(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:
        if self.m_metric != "aucprc":
            raise RuntimeError("Invalid metric for the evaluator. expected AUCPRC")

        return evaluate_auc(y_test, y_pred_proba, self.m_metric)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_estimator(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.Series,
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

    ev = Eval(metric)

    def eval_iteration(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
    ) -> float:
        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)

        return ev.score_proba(Y_test, preds)

    def sanitize_input(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for outcome in np.unique(Y_test):
            outcome_cnt = (np.asarray(Y_train) == outcome).sum()
            if outcome_cnt == 0:
                arr_filter = Y_test.values.ravel() != outcome

                X_test = X_test.loc[arr_filter]
                Y_test = Y_test.loc[arr_filter]

        for outcome in np.unique(Y_train):
            outcome_cnt = (np.asarray(Y_test) == outcome).sum()
            if outcome_cnt == 0:
                arr_filter = Y_train.values.ravel() != outcome

                X_train = X_train.loc[arr_filter]
                Y_train = Y_train.loc[arr_filter]

        return X_train, X_test, Y_train, Y_test

    indx = 0
    if n_folds == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed)

        X_train, X_test, Y_train, Y_test = sanitize_input(
            X_train, X_test, Y_train, Y_test
        )
        metric_[indx] = eval_iteration(X_train, X_test, Y_train, Y_test)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_index, test_index in skf.split(X, Y):

            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]
            X_test = X.loc[X.index[test_index]]
            Y_test = Y.loc[Y.index[test_index]]

            X_train, X_test, Y_train, Y_test = sanitize_input(
                X_train, X_test, Y_train, Y_test
            )

            metric_[indx] = eval_iteration(X_train, X_test, Y_train, Y_test)

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


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def score_classification_model(
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.Series,
    y_train: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    model = copy.deepcopy(estimator)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)


def evaluate_regression(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.Series,
    n_folds: int = 3,
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metrics = ["rmse", "wnd", "r2"]
    metrics_ = {}

    for m in metrics:
        metrics_[m] = np.zeros(n_folds)

    indx = 0

    def eval_iteration(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
    ) -> Tuple[float, float]:
        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        return (
            mean_squared_error(Y_test, preds),
            evaluate_wnd(preds, Y_test),
            r2_score(Y_test, preds),
        )

    if n_folds == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=seed)

        rmse, wnd, r2 = eval_iteration(X_train, X_test, Y_train, Y_test)
        metrics_["rmse"][indx] = rmse
        metrics_["wnd"][indx] = wnd
        metrics_["r2"][indx] = r2
    else:
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for train_index, test_index in skf.split(X, Y):

            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]
            X_test = X.loc[X.index[test_index]]
            Y_test = Y.loc[Y.index[test_index]]

            rmse, wnd, r2 = eval_iteration(X_train, X_test, Y_train, Y_test)
            metrics_["rmse"][indx] = rmse
            metrics_["wnd"][indx] = wnd
            metrics_["r2"][indx] = r2

            indx += 1

    output_clf_rmse = generate_score(metrics_["rmse"])
    output_clf_wnd = generate_score(metrics_["wnd"])
    output_clf_r2 = generate_score(metrics_["r2"])

    return {
        "clf": {
            "rmse": output_clf_rmse,
            "wnd": output_clf_wnd,
            "r2": output_clf_r2,
        },
        "str": {
            "rmse": print_score(output_clf_rmse),
            "wnd": print_score(output_clf_wnd),
            "r2": print_score(output_clf_r2),
        },
    }
