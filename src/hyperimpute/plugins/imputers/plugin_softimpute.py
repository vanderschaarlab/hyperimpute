# stdlib
import json
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.extmath import randomized_svd

# hyperimpute absolute
import hyperimpute.logger as log
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators
from hyperimpute.plugins.utils.metrics import RMSE

F32PREC = np.finfo(np.float32).eps


class SoftImpute(TransformerMixin):
    """The SoftImpute algorithm fits a low-rank matrix approximation to a matrix with missing values via nuclear-norm regularization. The algorithm can be used to impute quantitative data.
    To calibrate the the nuclear-norm regularization parameter(shrink_lambda), we perform cross-validation(_cv_softimpute)

    Args:
        maxit: int, default=500
            maximum number of imputation rounds to perform.
        convergence_threshold : float, default=1e-5
            Minimum ration difference between iterations before stopping.
        max_rank : int, default=2
            Perform a truncated SVD on each iteration with this value as its rank.
        shrink_lambda: float, default=0
            Value by which we shrink singular values on each iteration. If it's missing, it is calibrated using cross validation.
        cv_len: int, default=15
            the length of the grid on which the cross-validation is performed.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("softimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])


    Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices", by Mazumder, Hastie, and Tibshirani.
    """

    def __init__(
        self,
        maxit: int = 1000,
        convergence_threshold: float = 1e-5,
        max_rank: int = 2,
        shrink_lambda: float = 0,
        cv_len: int = 3,
        random_state: int = 0,
    ) -> None:
        self.cv_len = cv_len
        self.shrink_lambda = shrink_lambda
        self.maxit = maxit
        self.convergence_threshold = convergence_threshold
        self.max_rank = max_rank
        self.random_state = random_state

    def save(self) -> bytes:
        buff = {
            "cv_len": self.cv_len,
            "shrink_lambda": self.shrink_lambda,
            "maxit": self.maxit,
            "convergence_threshold": self.convergence_threshold,
            "max_rank": self.max_rank,
        }
        return json.dumps(buff).encode("utf-8")

    @classmethod
    def load(cls, buff: bytes) -> "SoftImpute":
        data = json.loads(buff.decode("utf-8"))
        return cls(
            maxit=data["maxit"],
            convergence_threshold=data["convergence_threshold"],
            max_rank=data["max_rank"],
            shrink_lambda=data["shrink_lambda"],
            cv_len=data["cv_len"],
        )

    @decorators.expect_ndarray_for(1)
    def fit(self, X: np.ndarray) -> "SoftImpute":
        """If the shrinkage step/lambda value is provided, it does nothing. Otherwise, it runs a cross-validation and approximates values for the shrinkage lambda.

        Args:
            X: np.ndarray
                Used for cross-validation, to calibrate the shrinkage lambda.

        Returns:
            self: The updated version, which a valid shrinkage step.
        """
        if self.shrink_lambda:
            return self

        self.shrink_lambda = self._approximate_shrink_val(X)

        return self

    @decorators.expect_ndarray_for(1)
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Imputes the provided dataset using the SoftImpute algorithm

        Args:
            X: np.ndarray
                A dataset with missing values.

        Returns:
            X_hat: The imputed dataset.
        """
        return self._softimpute(X, self.shrink_lambda)

    @decorators.expect_ndarray_for(1)
    def fit_transform(self, X: np.ndarray, **fit_params: Any) -> np.ndarray:
        return self.fit(X, **fit_params).transform(X)

    def _converged(self, Xold: np.ndarray, X: np.ndarray, mask: np.ndarray) -> bool:
        """Checks if the SoftImpute algorithm has converged.

        Args:
            Xold: np.ndarray
                The previous version of the imputed dataset.
            X: np.ndarray
                The new version of the imputed dataset.
            mask: np.ndarray
                The original missing mask.

        Returns:
            bool: True/False if the algorithm has converged.
        """
        rmse = RMSE(Xold, X, mask)
        denom = np.linalg.norm(Xold[mask])

        if denom == 0 or (denom < F32PREC and rmse > F32PREC):
            return False
        else:
            return (rmse / denom) < self.convergence_threshold

    def _svd(self, X: np.ndarray, shrink_val: float) -> np.ndarray:
        """Reconstructs X from low-rank thresholded SVD.

        Args:
            X: np.ndarray
                The previous version of the imputed dataset.
            shrink_val: float
                The value by which we shrink singular values on each iteration.

        Raises:
            RuntimeError: raised if the static checks on the final result fail.

        Returns:
            X_reconstructed: new candidate for the result.
        """
        if self.max_rank:
            U, s, V = randomized_svd(
                X, n_components=self.max_rank, random_state=self.random_state
            )
        else:
            U, s, V = np.linalg.svd(X, compute_uv=True, full_matrices=False)
        s_thresh = np.maximum(s - shrink_val, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        s_thresh = np.diag(s_thresh)
        X_reconstructed = np.dot(U_thresh, np.dot(s_thresh, V_thresh))

        if np.all(np.isnan(X_reconstructed)):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            log.critical(err)
            raise RuntimeError(err)

        return X_reconstructed

    def _softimpute(self, X: np.ndarray, shrink_val: float) -> np.ndarray:
        """Core loop of the algorithm. It approximates the imputed X using the SVD decomposition in a loop, until the algorithm converges/the maxit iteration is reached.

        Args:
            X: np.ndarray
                The previous version of the imputed dataset.
            shrink_val: float
                The value by which we shrink singular values on each iteration.

        Returns:
            X_hat: The imputed dataset.
        """
        mask = ~np.isnan(X)
        X_hat = X.copy()
        X_hat[~mask] = 0

        for i in range(self.maxit):
            X_reconstructed = self._svd(X_hat, shrink_val)
            if self._converged(X_hat, X_reconstructed, mask):
                log.debug("SoftImpute has converged after {i} iterations")
                break
            X_hat[~mask] = X_reconstructed[~mask]

        return X_hat

    def _approximate_shrink_val(self, X: np.ndarray) -> float:
        """Try to calibrate the shrinkage step using cross-validation. It simulates more missing items and tests the performance of different shrinkage values.

        Args:
            X: np.ndarray
                The dataset to use.

        Returns:
            float: The value to use for the shrinkage step.
        """
        mask = ~np.isnan(X)
        X0 = X.copy()
        X0[~mask] = 0

        # svd on x0
        if self.max_rank:
            _, s, _ = randomized_svd(X0, self.max_rank, random_state=self.random_state)
        else:
            s = np.linalg.svd(X0, compute_uv=False, full_matrices=False)

        lambda_max = np.max(s)
        lambda_min = 0.001 * lambda_max
        shrink_lambda = np.exp(
            np.linspace(np.log(lambda_min), np.log(lambda_max), self.cv_len).tolist()
        )

        X_test = self._simulate_more_nan(X, mask)
        cv_error = []

        for shrink_val in shrink_lambda:
            X_hat = self._softimpute(X_test, shrink_val)
            cv_error.append(np.sqrt(np.nanmean((X_hat.flatten() - X.flatten()) ** 2)))

        return shrink_lambda[np.argmin(cv_error)]

    def _simulate_more_nan(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate more missing values for cross-validation.

        Args:
            X: np.ndarray
                The dataset to use.
            mask: np.ndarray
                The existing missing positions

        Returns:
            Xsim: A new version of X with more missing values.
        """
        save_mask = mask.copy()
        for i in range(X.shape[0]):
            idx_obs = np.argwhere(save_mask[i, :] == 1).reshape((-1))
            if len(idx_obs) > 0:
                j = np.random.choice(idx_obs, 1)
                save_mask[i, j] = 0
        mmask = np.array(
            np.random.binomial(np.ones_like(save_mask), save_mask * 0.1), dtype=bool
        )
        Xsim = X.copy()
        Xsim[mmask] = np.nan
        return Xsim


class SoftImputePlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the SoftImpute strategy.

    Method:
        Details in the SoftImpute class implementation.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("softimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                      0             1             2             3
        0  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00
        1  3.820605e-16  1.708249e-16  1.708249e-16  3.820605e-16
        2  1.000000e+00  2.000000e+00  2.000000e+00  1.000000e+00
        3  2.000000e+00  2.000000e+00  2.000000e+00  2.000000e+00
    """

    def __init__(
        self,
        maxit: int = 1000,
        convergence_threshold: float = 1e-5,
        max_rank: int = 2,
        shrink_lambda: float = 0,
        cv_len: int = 3,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        self.maxit = maxit
        self.convergence_threshold = convergence_threshold
        self.max_rank = max_rank
        self.shrink_lambda = shrink_lambda
        self.cv_len = cv_len

        self._model = SoftImpute(random_state=random_state)

    @staticmethod
    def name() -> str:
        return "softimpute"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [params.Integer("max_rank", 2, 5), params.Float("shrink_lambda", 0, 10)]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SoftImputePlugin":
        self._model.fit(X.to_numpy())

        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X.to_numpy())


plugin = SoftImputePlugin
