# stdlib
from typing import Any, Callable, List, Optional

# third party
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers._hyperimpute_internals as internals
import hyperimpute.plugins.imputers.base as base
from hyperimpute.utils.distributions import enable_reproducible_results


class HyperImputePlugin(base.ImputerPlugin):
    """HyperImpute strategy.


    Args:
        classifier_seed: list.
            List of ClassifierPlugin names for the search pool.
        regression_seed: list.
            List of RegressionPlugin names for the search pool.
        imputation_order: int.
            0 - ascending, 1 - descending, 2 - random
        baseline_imputer: int.
            0 - mean, 1 - median, 2- most_frequent
        optimizer: str.
            Hyperparam search strategy. Options: simple, hyperband, bayesian
        class_threshold: int.
            Maximum number of unique items in a categorical column.
        optimize_thresh: int.
            The number of subsamples used for the model search.
        n_inner_iter: int.
            number of imputation iterations.
        select_model_by_column: bool.
            If False, reuse the first model selected in the current iteration for all columns. Else, search the model for each column.
        select_model_by_iteration: bool.
            If False, reuse the models selected in the first iteration. Otherwise, refresh the models on each iteration.
        select_lazy: bool.
            If True, if there is a trend towards a certain model architecture, the loop reuses than for all columns, instead of calling the optimizer.
        inner_loop_hook: Callable.
            Debug hook, called before each iteration.
        random_state: int.
            random seed.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("hyperimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])


    Reference: "HyperImpute: Generalized Iterative Imputation with Automatic Model Selection"
    """

    initial_strategy_vals = ["mean", "median", "most_frequent"]
    imputation_order_vals = ["ascending", "descending", "random"]

    def __init__(
        self,
        classifier_seed: list = internals.LARGE_DATA_CLF_SEEDS,
        regression_seed: list = internals.LARGE_DATA_REG_SEEDS,
        imputation_order: int = 2,  # imputation_order_vals
        baseline_imputer: int = 0,  # initial_strategy_vals
        optimizer: str = "simple",
        class_threshold: int = 5,
        optimize_thresh: int = 5000,
        n_inner_iter: int = 40,
        random_state: int = 0,
        select_model_by_column: bool = True,
        select_model_by_iteration: bool = True,
        select_patience: int = 5,
        select_lazy: bool = True,
        inner_loop_hook: Optional[Callable] = None,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)
        self.classifier_seed = classifier_seed
        self.regression_seed = regression_seed
        self.imputation_order = imputation_order
        self.baseline_imputer = baseline_imputer
        self.optimizer = optimizer
        self.class_threshold = class_threshold
        self.optimize_thresh = optimize_thresh
        self.n_inner_iter = n_inner_iter
        self.random_state = random_state
        self.select_model_by_column = select_model_by_column
        self.select_model_by_iteration = select_model_by_iteration
        self.select_patience = select_patience
        self.select_lazy = select_lazy
        self.inner_loop_hook = inner_loop_hook

        self.model = internals.IterativeErrorCorrection(
            "hyperimpute_plugin",
            classifier_seed=self.classifier_seed,
            regression_seed=self.regression_seed,
            optimizer=self.optimizer,
            baseline_imputer=HyperImputePlugin.initial_strategy_vals[
                self.baseline_imputer
            ],
            imputation_order_strategy=HyperImputePlugin.imputation_order_vals[
                self.imputation_order
            ],
            class_threshold=self.class_threshold,
            optimize_thresh=self.optimize_thresh,
            n_inner_iter=self.n_inner_iter,
            select_model_by_column=self.select_model_by_column,
            select_model_by_iteration=self.select_model_by_iteration,
            select_patience=self.select_patience,
            select_lazy=self.select_lazy,
            inner_loop_hook=self.inner_loop_hook,
            random_state=random_state,
        )

    @staticmethod
    def name() -> str:
        return "hyperimpute"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "HyperImputePlugin":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.fit_transform(X)

    def models(self) -> dict:
        return self.model.models()

    def trace(self) -> dict:
        return {
            "objective": self.model.perf_trace,
            "models": self.model.model_trace,
        }


plugin = HyperImputePlugin
