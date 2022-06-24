# stdlib
from functools import reduce
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

# hyperimpute absolute
import hyperimpute.logger as log
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators


class EM(TransformerMixin):
    """The EM algorithm is an optimization algorithm that assumes a distribution for the partially missing data and tries to maximize the expected complete data log-likelihood under that distribution.

    Steps:
        1. For an input dataset X with missing values, we assume that the values are sampled from distribution N(Mu, Sigma).
        2. We generate the "observed" and "missing" masks from X, and choose some initial values for Mu = Mu0 and Sigma = Sigma0.
        3. The EM loop tries to approximate the (Mu, Sigma) pair by some iterative means under the conditional distribution of missing components.
        4. The E step finds the conditional expectation of the "missing" data, given the observed values and current estimates of the parameters. These expectations are then substituted for the "missing" data.
        5. In the M step, maximum likelihood estimates of the parameters are computed as though the missing data had been filled in.
        6. The X_reconstructed contains the approximation after each iteration.

    Args:
        maxit: int, default=500
            maximum number of imputation rounds to perform.
        convergence_threshold : float, default=1e-08
            Minimum ration difference between iterations before stopping.

    Paper: "Maximum Likelihood from Incomplete Data via the EM Algorithm", A. P. Dempster, N. M. Laird and D. B. Rubin
    """

    def __init__(self, maxit: int = 500, convergence_threshold: float = 1e-08) -> None:
        self.maxit = maxit
        self.convergence_threshold = convergence_threshold

    @decorators.expect_ndarray_for(1)
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Imputes the provided dataset using the EM strategy.

        Args:
            X: np.ndarray
                A dataset with missing values.

        Returns:
            Xhat: The imputed dataset.
        """
        return self._impute_em(X)

    def _converged(
        self,
        Mu: np.ndarray,
        Sigma: np.ndarray,
        Mu_new: np.ndarray,
        Sigma_new: np.ndarray,
    ) -> bool:
        """Checks if the EM loop has converged.

        Args:
            Mu: np.ndarray
                The previous value of the mean.
            Sigma: np.ndarray
                The previous value of the variance.
            Mu_new: np.ndarray
                The new value of the mean.
            Sigma_new: np.ndarray
                The new value of the variance.

        Returns:
            bool: True/False if the algorithm has converged.
        """

        return (
            np.linalg.norm(Mu - Mu_new) < self.convergence_threshold
            and np.linalg.norm(Sigma - Sigma_new, ord=2) < self.convergence_threshold
        )

    def _em(
        self,
        X_reconstructed: np.ndarray,
        Mu: np.ndarray,
        Sigma: np.ndarray,
        observed: np.ndarray,
        missing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The EM step.

        Args:
            X_reconstructed: np.ndarray
                The current imputation approximation.
            Mu: np.ndarray
                The previous value of the mean.
            Sigma: np.ndarray
                The previous value of the variance.
            observed: np.ndarray
                Mask of the observed values in the original input.
            missing: np.ndarray
                Mask of the missing values in the original input.

        Returns:
            ndarray: The new approximation of the mean.
            ndarray: The new approximation of the variance.
            ndarray: The new imputed dataset.

        """
        rows, columns = X_reconstructed.shape

        one_to_nc = np.arange(1, columns + 1, step=1)
        Mu_tilde, Sigma_tilde = {}, {}

        for i in range(rows):
            Sigma_tilde[i] = np.zeros(columns**2).reshape(columns, columns)
            if set(observed[i, :]) == set(one_to_nc - 1):
                # nothing to impute
                continue

            missing_i = missing[i, :][missing[i, :] != -1]
            observed_i = observed[i, :][observed[i, :] != -1]
            S_MM = Sigma[np.ix_(missing_i, missing_i)]
            S_MO = Sigma[np.ix_(missing_i, observed_i)]
            S_OM = S_MO.T
            S_OO = Sigma[np.ix_(observed_i, observed_i)]
            Mu_tilde[i] = Mu[np.ix_(missing_i)] + S_MO @ np.linalg.inv(S_OO) @ (
                X_reconstructed[i, observed_i] - Mu[np.ix_(observed_i)]
            )
            X_reconstructed[i, missing_i] = Mu_tilde[i]
            S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
            Sigma_tilde[i][np.ix_(missing_i, missing_i)] = S_MM_O

        Mu_new = np.mean(X_reconstructed, axis=0)
        Sigma_new = (
            np.cov(X_reconstructed.T, bias=1)
            + reduce(np.add, Sigma_tilde.values()) / rows
        )

        return Mu_new, Sigma_new, X_reconstructed

    def _impute_em(self, X: np.ndarray) -> np.ndarray:
        """The EM imputation core loop.

        Args:
            X: np.ndarray
                The dataset with missing values.

        Raises:
            RuntimeError: raised if the static checks on the final result fail.

        Returns:
            ndarray: The dataset with imputed values.
        """
        rows, columns = X.shape
        mask = ~np.isnan(X)

        one_to_nc = np.arange(1, columns + 1, step=1)
        missing = one_to_nc * (~mask) - 1
        observed = one_to_nc * mask - 1

        Mu = np.nanmean(X, axis=0)
        observed_rows = np.where(np.isnan(sum(X.T)) is False)[0]
        Sigma = np.cov(
            X[
                observed_rows,
            ].T
        )
        if np.isnan(Sigma).any():
            Sigma = np.diag(np.nanvar(X, axis=0))

        X_reconstructed = X.copy()

        for iteration in range(self.maxit):
            try:
                Mu_new, Sigma_new, X_reconstructed = self._em(
                    X_reconstructed, Mu, Sigma, observed, missing
                )

                if self._converged(Mu, Sigma, Mu_new, Sigma_new):
                    log.debug(f"EM converged after {iteration} iterations.")
                    break

                Mu = Mu_new
                Sigma = Sigma_new
            except BaseException as e:
                log.critical(f"EM step failed. {e}")
                break

        if np.all(np.isnan(X_reconstructed)):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            log.critical(err)
            raise RuntimeError(err)

        return X_reconstructed


class EMPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the EM strategy.

    The EM algorithm is an optimization algorithm that assumes a distribution for the partially missing data and tries to maximize the expected complete data log-likelihood under that distribution.

    Steps:
        1. For an input dataset X with missing values, we assume that the values are sampled from distribution N(Mu, Sigma).
        2. We generate the "observed" and "missing" masks from X, and choose some initial values for Mu = Mu0 and Sigma = Sigma0.
        3. The EM loop tries to approximate the (Mu, Sigma) pair by some iterative means under the conditional distribution of missing components.
        4. The E step finds the conditional expectation of the "missing" data, given the observed values and current estimates of the parameters. These expectations are then substituted for the "missing" data.
        5. In the M step, maximum likelihood estimates of the parameters are computed as though the missing data had been filled in.
        6. The X_reconstructed contains the approximation after each iteration.

    Args:
        maxit: int, default=500
            maximum number of imputation rounds to perform.
        convergence_threshold : float, default=1e-08
            Minimum ration difference between iterations before stopping.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("EM")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])

    Reference: "Maximum Likelihood from Incomplete Data via the EM Algorithm", A. P. Dempster, N. M. Laird and D. B. Rubin
    """

    def __init__(
        self,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        self._model = EM()

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "EMPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X.to_numpy())

    @staticmethod
    def name() -> str:
        return "EM"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("maxit", 100, 800, 100),
            params.Categorical("convergence_threshold", [1e-08, 1e-07, 1e-06]),
        ]


plugin = EMPlugin
