# third party
import numpy as np


def MAE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        MAE : np.ndarray
    """
    mask_ = mask.astype(bool)
    return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def RMSE(X: np.ndarray, X_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth

    Args:
        X : Data with imputed variables.
        X_true : Ground truth.
        mask : Missing value mask (missing if True)

    Returns:
        RMSE : np.ndarray

    """
    mask_ = mask.astype(bool)
    return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())


__all__ = ["MAE", "RMSE"]
