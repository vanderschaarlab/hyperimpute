# stdlib
from typing import Tuple, Union

# third party
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# hyperimpute absolute
import hyperimpute.logger as log


def get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
    if isinstance(y_pred_proba, tuple):
        y_pred_proba_tmp = y_pred_proba[1]
    elif nclasses <= 2 and isinstance(y_pred_proba, (np.ndarray, np.generic)):
        y_pred_proba_tmp = (
            y_pred_proba if len(y_pred_proba.shape) < 2 else y_pred_proba[:, 1]
        )
    else:
        y_pred_proba_tmp = y_pred_proba
    return y_pred_proba_tmp


def evaluate_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "aucroc",
    classes: Union[np.ndarray, None] = None,
) -> float:

    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = len(set(np.ravel(y_test)))

    y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    if n_classes > 2:

        log.debug(f"+evaluate_auc {y_test.shape} {y_pred_proba_tmp.shape}")

        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_auc: dict = dict()

        if classes is None:
            classes = sorted(set(np.ravel(y_test)))
            log.debug(
                "warning: classes is none and more than two "
                " (#{}), classes assumed to be an ordered set:{}".format(
                    n_classes, classes
                )
            )

        y_test = label_binarize(y_test, classes=classes)

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_test, y_pred_proba_tmp, average="micro"
        )

        if metric == "aucroc":
            return roc_auc["micro"]
        elif metric == "aucprc":
            return average_precision["micro"]
        else:
            raise RuntimeError(f"invalid evaluation metric {metric}")
    else:
        if metric == "aucroc":
            return roc_auc_score(np.ravel(y_test), y_pred_proba_tmp)
        elif metric == "aucprc":
            return average_precision_score(np.ravel(y_test), y_pred_proba_tmp)
        else:
            raise RuntimeError(f"invalid evaluation metric {metric}")


def evaluate_wnd(imputed: pd.DataFrame, ground: pd.DataFrame) -> pd.DataFrame:
    res = 0
    for col in range(ground.shape[1]):
        res += wasserstein_distance(
            np.asarray(ground)[:, col], np.asarray(imputed)[:, col]
        )
    return res


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))
