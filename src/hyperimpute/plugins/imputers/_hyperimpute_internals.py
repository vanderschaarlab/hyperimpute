# stdlib
import copy
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# third party
import numpy as np
import optuna
import pandas as pd
from pydantic import validate_arguments
from sklearn.impute import MissingIndicator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch

# hyperimpute absolute
import hyperimpute.logger as log
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.prediction import PredictionPlugin, Predictions
from hyperimpute.utils.optimizer import EarlyStoppingExceeded, create_study
from hyperimpute.utils.serialization import Serializable
from hyperimpute.utils.tester import evaluate_estimator, evaluate_regression

INNER_TOL = 1e-8
OUTER_TOL = 1e-3

SMALL_DATA_CLF_SEEDS = [
    "random_forest",
    "logistic_regression",
]
SMALL_DATA_REG_SEEDS = [
    "random_forest_regressor",
    "linear_regression",
]

LARGE_DATA_CLF_SEEDS = SMALL_DATA_CLF_SEEDS + [
    "xgboost",
    "catboost",
]
LARGE_DATA_REG_SEEDS = SMALL_DATA_REG_SEEDS + [
    "xgboost_regressor",
    "catboost_regressor",
]
if torch.cuda.is_available():
    LARGE_DATA_CLF_SEEDS.append("neural_nets")
    LARGE_DATA_REG_SEEDS.append("neural_nets_regression")


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class HyperbandOptimizer(Serializable):
    """Optimization helper based on HyperBand.

    Args:
        name: str
            ID
        category: str
            classification/regression. Impacts the objective function
        classifier_seed: list
            List of classification methods to evaluate. Used when category = "classifier"
        regression_seed: list
            List of regression methods to evaluate. Used when category = "regression"
        max_iter: int
            maximum iterations per configuration
        eta: int
            configuration downsampling rate
        random_state: int
            random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
        max_iter: int = 27,  # maximum iterations per configuration
        eta: int = 3,  # defines configuration downsampling rate (default = 3)
        random_state: int = 0,
    ) -> None:
        super().__init__()

        self.name = name
        self.category = category
        self.failure_score = -9999999
        self.random_state = random_state

        self.predictions = Predictions(category=category)
        if category == "classifier":
            self.seeds = classifier_seed
        else:
            self.seeds = regression_seed

        self.max_iter = max_iter
        self.eta = eta

        def logeta(x: Any) -> Any:
            return math.log(x) / math.log(self.eta)

        self.logeta = logeta
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self._reset()

        self.model_best_score = {}
        for seed in self.seeds:
            self.model_best_score[seed] = -np.inf

        self.candidate = {
            "score": self.failure_score,
            "name": self.seeds[0],
            "params": {},
        }

    def _reset(self) -> None:
        self.visited: Set[str] = set()

    def _hash_dict(self, name: str, dict_val: dict) -> str:
        return json.dumps(
            {"name": name, "val": dict_val}, sort_keys=True, cls=NpEncoder
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _sample_model(self, name: str, n: int) -> list:
        hashed = self._hash_dict(name, {})
        result: List[Tuple] = []
        if hashed not in self.visited:
            self.visited.add(hashed)
            result.append((name, {}))
            n -= 1

        for i in range(n):
            params = self.predictions.get_type(name).sample_hyperparameters_np()
            hashed = self._hash_dict(name, params)

            if hashed in self.visited:
                continue

            self.visited.add(hashed)
            result.append((name, params))

        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _sample(self, n: int) -> list:
        results = []
        for name in self.seeds:
            results.extend(self._sample_model(name, n))
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _baseline(self, X: pd.DataFrame, y: pd.Series) -> None:
        for seed in self.seeds:
            self._eval_params(seed, X, y, hyperparam_search_iterations=1)
        # TODO: balance methods

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _eval_params(
        self, model_name: str, X: pd.DataFrame, y: pd.Series, **params: Any
    ) -> float:
        model = self.predictions.get(model_name, **params)
        for n_folds in [2, 1]:
            try:
                if self.category == "regression":
                    out = evaluate_regression(model, X, y, n_folds=n_folds)
                    score = out["clf"]["r2"][0]
                else:
                    out = evaluate_estimator(model, X, y, n_folds=n_folds)
                    score = out["clf"]["aucroc"][0]
                break
            except BaseException as e:
                log.error(f"      >>> {self.name}:{model_name}: eval failed {e}")
                score = self.failure_score

        if score > self.candidate["score"]:
            self.candidate = {
                "score": score,
                "params": params,
                "name": model_name,
            }
        if score > self.model_best_score[model_name]:
            self.model_best_score[model_name] = score

        return score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[PredictionPlugin, float]:
        self._reset()
        self._baseline(X, y)

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta**s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = self._sample(math.ceil(n / len(self.seeds)))

            for i in range(s + 1):  # changed from s + 1
                if len(T) == 0:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = int(math.ceil(n * self.eta ** (-i)))
                n_iterations = r * self.eta ** (i)

                scores = []

                for model_name, model_params in T:
                    score = self._eval_params(
                        model_name,
                        X,
                        y,
                        hyperparam_search_iterations=n_iterations,
                        random_state=self.random_state,
                        **model_params,
                    )
                    scores.append(score)
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                saved = int(math.ceil(n_configs / self.eta))

                indices = np.argsort(scores)
                T = [T[i] for i in indices]
                T = T[-saved:]
                scores = [scores[i] for i in indices]
                scores = scores[-saved:]

        log.info(
            f"      >>> {self.name} -- best candidate {self.candidate['name']}: ({self.candidate['params']}) --- score : {self.candidate['score']}"
        )

        self.seeds = sorted(
            self.model_best_score, key=self.model_best_score.get, reverse=True  # type: ignore
        )[:2]

        return (
            Predictions(category=self.category).get(
                self.candidate["name"], **self.candidate["params"]
            ),
            self.candidate["score"],
        )


class BayesianOptimizer(Serializable):
    """Optimization helper based on Bayesian Optimization.

    Args:
        name: str
            ID
        category: str
            classification/regression. Impacts the objective function
        classifier_seed: list
            List of classification methods to evaluate. Used when category = "classifier"
        regression_seed: list
            List of regression methods to evaluate. Used when category = "regression"
        patience: int
            maximum iterations without any gain
        random_state: int
            random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
        patience: int = 10,
        inner_patience: int = 5,
        random_state: int = 0,
    ):
        super().__init__()

        self.name = name
        self.category = category
        self.random_state = random_state

        self.failure_score = -9999999
        if category == "classifier":
            self.seeds = classifier_seed
        else:
            self.seeds = regression_seed

        self.bo_studies = {}
        self.patience = patience
        self.inner_patience = inner_patience

        for seed in self.seeds:
            self.bo_studies[seed] = create_study(
                study_name=f"{self.name}_imputer_evaluation_{seed}",
                direction="maximize",
                load_if_exists=False,
                patience=self.patience,
            )

        self.best_score = self.failure_score
        self.best_candidate = self.seeds[0]
        self.best_params: dict = {}

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_plugin_args(
        self,
        plugin_name: str,
        plugin_args: dict,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        plugin = Predictions(category=self.category).get_type(plugin_name)

        model = plugin(random_state=self.random_state, **plugin_args)
        for n_folds in [2, 1]:
            try:
                if self.category == "regression":
                    out = evaluate_regression(model, X, y, n_folds=n_folds)
                    score = out["clf"]["r2"][0]
                else:
                    out = evaluate_estimator(model, X, y, n_folds=n_folds)
                    score = out["clf"]["aucroc"][0]
                break
            except BaseException as e:
                log.error(f"failed to evaluate {plugin.name()} error {e}")
                score = self.failure_score

        return score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_plugin(
        self,
        plugin_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        prev_best_score: float,
    ) -> tuple:
        # BO evaluation for a single plugin
        plugin = Predictions(category=self.category).get_type(plugin_name)

        early_stopping_patience: int = 0
        study, pruner = self.bo_studies[plugin_name]

        def evaluate_args(**kwargs: Any) -> float:
            return self.evaluate_plugin_args(plugin_name, kwargs, X, y)

        baseline_score = evaluate_args()
        pruner.report_score(baseline_score)

        def objective(trial: optuna.Trial) -> float:
            nonlocal early_stopping_patience
            args = plugin.sample_hyperparameters(trial)
            pruner.check_trial(trial)

            score = evaluate_args(**args)

            pruner.report_score(score)

            if score < self.best_score or score < baseline_score:
                early_stopping_patience += 1
            else:
                early_stopping_patience = 0

            if early_stopping_patience >= self.inner_patience:
                raise EarlyStoppingExceeded

            return score

        try:
            study.optimize(objective, n_trials=self.patience - 1, timeout=60 * 3)
        except EarlyStoppingExceeded:
            pass

        if baseline_score > study.best_value:
            return baseline_score, {}

        return study.best_value, study.best_trial.params

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[PredictionPlugin, float]:

        old_best = self.best_score
        self.best_score = self.evaluate_plugin_args(
            self.best_candidate, self.best_params, X_train, y_train
        )
        best_score = self.best_score
        log.info(
            f"[Evaluate {self.name}] previous config new score = {self.best_score}. old score = {old_best}"
        )

        if self.category == "classifier":
            mapped_labels = sorted(y_train.unique())
            mapping = {}
            for idx, label in enumerate(mapped_labels):
                mapping[label] = idx
            y_train = y_train.map(mapping)

        for plugin_name in self.seeds:
            new_score, new_params = self.evaluate_plugin(
                plugin_name, X_train, y_train, best_score
            )
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_candidate = plugin_name
                self.best_params = new_params

        log.info(
            f"     >>> Column {self.name} <-- score {self.best_score} <-- Model {self.best_candidate}({self.best_params})"
        )
        return (
            Predictions(category=self.category).get(
                self.best_candidate, **self.best_params
            ),
            self.best_score,
        )


class SimpleOptimizer(Serializable):
    """Optimization helper based on default args evaluation.

    Args:
        name: str
            ID
        category: str
            classification/regression. Impacts the objective function
        classifier_seed: list
            List of classification methods to evaluate. Used when category = "classifier"
        regression_seed: list
            List of regression methods to evaluate. Used when category = "regression"
        random_state: int
            random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
        random_state: int = 0,
    ) -> None:
        super().__init__()

        self.name = name
        self.category = category
        self.random_state = random_state

        self.failure_score = -9999999
        self.classifier_seed = classifier_seed
        self.regression_seed = regression_seed
        if category == "classifier":
            self.seeds = classifier_seed
        else:
            self.seeds = regression_seed

        self.candidate = {
            "score": self.failure_score,
            "name": self.seeds[0],
            "params": {},
        }

        self.predictions = Predictions(category=category)

        self.model_best_score = {}
        for seed in self.seeds:
            self.model_best_score[seed] = -np.inf

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _eval_params(
        self, model_name: str, X: pd.DataFrame, y: pd.Series, **params: Any
    ) -> float:
        model = self.predictions.get(model_name, **params)
        for n_folds in [2, 1]:
            try:
                if self.category == "regression":
                    out = evaluate_regression(model, X, y, n_folds=n_folds)
                    score = out["clf"]["r2"][0]
                else:
                    out = evaluate_estimator(model, X, y, n_folds=n_folds)
                    score = out["clf"]["aucroc"][0]
                break
            except BaseException as e:
                log.error(
                    f"      >>> {self.name}:{model_name}:{n_folds} folds eval failed {e}"
                )
                score = self.failure_score

        if score > self.candidate["score"]:
            self.candidate = {
                "score": score,
                "params": params,
                "name": model_name,
            }
            if score > self.model_best_score[model_name]:
                self.model_best_score[model_name] = score

        return score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[PredictionPlugin, float]:
        for seed in self.seeds:
            self._eval_params(seed, X, y, random_state=self.random_state)
        log.info(
            f"     >>> Column {self.name} <-- score {self.candidate['score']} <-- Model {self.candidate['name']}"
        )
        return (
            Predictions(category=self.category).get(
                self.candidate["name"], **self.candidate["params"]
            ),
            self.candidate["score"],
        )


class IterativeErrorCorrection(Serializable):
    """HyperImpute core runner.

    Args:
        study: str
            ID
        classifier_seed: list
            List of classification methods to evaluate. Used when category = "classifier"
        regression_seed: list
            List of regression methods to evaluate. Used when category = "regression"
        optimizer: str
            Which hyperparam search strategy to use: hyperband, bayesian, simple.
        baseline_imputer: str
            baseline imputation method.
        class_threshold: int
            Maximum unique values for categoricals.
        optimize_thresh: int
            How many samples to use for hyperparam search
        optimize_thresh_upper: int
            If the available data is bigger than this value, use just the SimpleOptimizer.
        n_inner_iter: int
            Maximum number of imputation iterations.
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
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study: str,
        classifier_seed: list,
        regression_seed: list,
        optimizer: str,
        baseline_imputer: str = "mean",
        class_threshold: int = 20,
        optimize_thresh: int = 1000,
        optimize_thresh_upper: int = 3000,
        imputation_order_strategy: str = "ascending",
        n_inner_iter: int = 50,
        n_min_inner_iter: int = 10,
        select_model_by_column: bool = True,
        select_model_by_iteration: bool = True,
        select_patience: int = 3,
        select_lazy: bool = True,
        inner_loop_hook: Optional[Callable] = None,
        random_state: int = 0,
    ):
        super().__init__()

        if optimizer not in [
            "hyperband",
            "bayesian",
            "simple",
        ]:
            raise RuntimeError(f"Invalid optimizer {optimizer}")

        self.select_model_by_column = select_model_by_column
        self.select_model_by_iteration = select_model_by_iteration
        self.select_patience = select_patience
        self.select_lazy = select_lazy
        self.random_state = random_state

        if not select_model_by_column:
            class_threshold = 0

        log.info(
            f"Iteration imputation: select_model_by_column: {select_model_by_column}, select_model_by_iteration: {select_model_by_iteration}"
        )
        self.study = study
        self.class_threshold = class_threshold
        self.baseline_imputer = Imputers().get(baseline_imputer)
        self.optimize_thresh = optimize_thresh
        self.optimize_thresh_upper = optimize_thresh_upper
        self.n_inner_iter = n_inner_iter
        self.n_min_inner_iter = n_min_inner_iter
        self.classifier_seed = classifier_seed
        self.regression_seed = regression_seed
        self.imputation_order_strategy = imputation_order_strategy
        self.inner_loop_hook = inner_loop_hook

        self.optimizer: Any
        if optimizer == "hyperband":
            self.optimizer = HyperbandOptimizer
        elif optimizer == "bayesian":
            self.optimizer = BayesianOptimizer
        else:
            self.optimizer = SimpleOptimizer

        self.avail_data_thresh = 500
        self.perf_trace: Dict[str, list] = {}
        self.model_trace: Dict[str, list] = {}

    def _select_seeds(self, miss_cnt: int) -> dict:
        clf = self.classifier_seed
        reg = self.regression_seed

        def intersect_or_right(left: list, right: list) -> list:
            res = list(set(left) & set(right))
            if len(res) == 0:
                res = right

            return res

        if miss_cnt < self.avail_data_thresh:
            clf = intersect_or_right(clf, SMALL_DATA_CLF_SEEDS)
            reg = intersect_or_right(reg, SMALL_DATA_REG_SEEDS)

        return {
            "classifier_seed": clf,
            "regression_seed": reg,
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _setup(self, X: pd.DataFrame) -> pd.DataFrame:
        # Encode the categorical columns
        # Reset internal caches
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        X.columns = X.columns.map(str)

        self.mask = self._missing_indicator(X)

        self.all_cols = X.columns
        self.categorical_cols = []
        self.continuous_cols = []

        self.column_to_model: dict = {}
        self.column_to_optimizer: dict = {}

        # TODO: add other strategies
        self.imputation_order = list(self.all_cols)

        self.encoders = {}

        for col in X.columns:
            avail_cnt = X[col].notna().sum()
            optimizer = self.optimizer
            if avail_cnt > self.optimize_thresh_upper:
                optimizer = SimpleOptimizer

            if self._is_categorical(X, col):
                self.categorical_cols.append(col)

                self.column_to_optimizer[col] = optimizer(
                    col,
                    "classifier",
                    random_state=self.random_state,
                    **self._select_seeds(avail_cnt),
                )
            else:
                self.continuous_cols.append(col)
                self.column_to_optimizer[col] = optimizer(
                    col,
                    "regression",
                    random_state=self.random_state,
                    **self._select_seeds(avail_cnt),
                )

            if X[col].dtype == "object" or self._is_categorical(X, col):
                # TODOD: One hot encoding for smaller cnt

                existing_vals = X[col][X[col].notnull()]

                le = LabelEncoder()
                X.loc[X[col].notnull(), col] = le.fit_transform(existing_vals).astype(
                    int
                )
                self.encoders[col] = le

        self.limits = {}
        for col in X.columns:
            self.limits[col] = (X[col].min(), X[col].max())

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _tear_down(self, X: pd.DataFrame) -> pd.DataFrame:
        # Revert the encoding after processing the data
        for col in self.encoders:
            X[col] = self.encoders[col].inverse_transform(X[col].astype(int))

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _get_neighbors_for_col(self, col: str) -> list:
        covs = list(self.all_cols)
        covs.remove(col)

        return covs

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_same_type(self, lhs: str, rhs: str) -> bool:
        ltype = lhs in self.categorical_cols
        rtype = rhs in self.categorical_cols

        return ltype == rtype

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _check_similar(self, X: pd.DataFrame, col: str) -> Any:
        if not self.select_lazy:
            return None

        similar_cols = []
        for ref_col in self.column_to_model:
            if not self.select_model_by_column:
                return copy.deepcopy(self.column_to_model[ref_col])

            if not self._is_same_type(ref_col, col):
                continue

            arch = self.column_to_model[ref_col].name()

            if arch in similar_cols:
                return copy.deepcopy(self.column_to_model[ref_col])

            similar_cols.append(self.column_to_model[ref_col].name())

        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _optimize_model_for_column(self, X: pd.DataFrame, col: str) -> float:
        # BO evaluation for a single column
        if self.mask[col].sum() == 0:
            return 0

        similar_candidate = self._check_similar(X, col)
        if similar_candidate is not None:
            self.column_to_model[col] = similar_candidate
            return 0

        cov_cols = self._get_neighbors_for_col(col)
        covs = X[cov_cols]

        target = X[col]

        if self.mask[col].sum() == 0:
            X_train = covs
            y_train = target
        else:
            X_train = covs[~self.mask[col]]
            y_train = target[~self.mask[col]]

        if self.optimize_thresh < len(X_train):
            X_train = X_train.sample(self.optimize_thresh)
            y_train = y_train[X_train.index]

        if col in self.categorical_cols:
            y_train = y_train.astype(int)

        candidate, score = self.column_to_optimizer[col].evaluate(X_train, y_train)
        self.column_to_model[col] = candidate
        self.perf_trace.setdefault(col, []).append(score)
        self.model_trace.setdefault(col, []).append(candidate.name())

        return score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _optimize(self, X: pd.DataFrame) -> float:
        # BO evaluation to select the best models for each columns
        if self.select_model_by_iteration:
            self.column_to_model = {}

        iteration_score: float = 0
        for col in self.imputation_order:
            iteration_score += self._optimize_model_for_column(X, col)

        return iteration_score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _impute_single_column(
        self, X: pd.DataFrame, col: str, train: bool
    ) -> pd.DataFrame:
        # Run an iteration of imputation on a column
        if self.mask[col].sum() == 0:
            return X

        cov_cols = self._get_neighbors_for_col(col)
        covs = X[cov_cols]

        target = X[col]

        X_train = covs[~self.mask[col]]
        y_train = target[~self.mask[col]]

        if col in self.categorical_cols:
            y_train = y_train.astype(int)

        if len(np.unique(y_train)) == 1:
            X[col][self.mask[col]] = np.asarray(y_train)[0]
            return X

        est = self.column_to_model[col]

        if train:
            est.fit(X_train, y_train)

        X[col][self.mask[col]] = est.predict(covs[self.mask[col]]).values.squeeze()

        col_min, col_max = self.limits[col]
        X[col][self.mask[col]] = np.clip(X[col][self.mask[col]], col_min, col_max)

        return X

    def models(self) -> dict:
        return self.column_to_model

    def _get_imputation_order(self) -> list:
        if self.imputation_order_strategy == "ascending":
            return self.imputation_order
        elif self.imputation_order_strategy == "descending":
            return list(reversed(self.imputation_order))
        else:
            random.shuffle(self.imputation_order)
            return self.imputation_order

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _initial_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        # Use baseline imputer for initial values
        return self.baseline_imputer.fit_transform(X)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _is_categorical(self, X: pd.DataFrame, col: str) -> bool:
        # Helper for filtering categorical columns
        return len(X[col].unique()) < self.class_threshold

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _missing_indicator(self, X: pd.DataFrame) -> pd.DataFrame:
        # Helper for generating missingness mask
        return pd.DataFrame(
            MissingIndicator(features="all").fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _fit_transform_inner_optimization(self, X: pd.DataFrame) -> pd.DataFrame:
        log.info("  > HyperImpute using inner optimization")
        best_obj_score = -10e10
        patience = 0

        for it in range(self.n_inner_iter):
            log.info(f"  > Imputation iter {it}")
            if self.select_model_by_iteration:
                self.column_to_model = {}

            obj_score: float = 0

            if self.inner_loop_hook:
                self.inner_loop_hook(it, self._tear_down(X.copy()))

            X_prev = X.copy()

            cols = self._get_imputation_order()
            for col in cols:
                obj_score += self._optimize_model_for_column(X, col)
                X = self._impute_single_column(X.copy(), col, True)

            inf_norm = np.linalg.norm(X - X_prev, ord=np.inf, axis=None)
            if inf_norm < INNER_TOL and it > self.n_min_inner_iter:
                log.info(
                    f"     >>>> Early stopping on imputation diff iteration : {it} err: {mean_squared_error(X_prev, X)}"
                )
                break

            if obj_score > best_obj_score:
                best_obj_score = obj_score
                patience = 0
            else:
                patience += 1

            if patience > self.select_patience:
                log.info("     >>>> Early stopping on objective diff iteration")
                break

        return X

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Run imputation
        X = self._setup(X)

        Xt_init = self._initial_imputation(X)
        Xt_init.columns = X.columns

        Xt = Xt_init.copy()

        Xt = self._fit_transform_inner_optimization(Xt)

        return self._tear_down(Xt)
