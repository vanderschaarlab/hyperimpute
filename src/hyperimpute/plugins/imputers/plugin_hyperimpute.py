# stdlib
import copy
import json
import math
import random
from typing import Any, List, Set, Tuple

# third party
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# hyperimpute absolute
import hyperimpute.logger as log
import hyperimpute.plugins.core.params as params
from hyperimpute.plugins.imputers import Imputers
import hyperimpute.plugins.imputers.base as base
from hyperimpute.plugins.prediction import Predictions
from hyperimpute.utils.optimizer import EarlyStoppingExceeded, create_study
from hyperimpute.utils.tester import evaluate_estimator, evaluate_regression

TOL = 1e-3

SMALL_DATA_CLF_SEEDS = ["logistic_regression", "random_forest"]
SMALL_DATA_REG_SEEDS = ["linear_regression", "random_forest_regressor"]

LARGE_DATA_CLF_SEEDS = SMALL_DATA_CLF_SEEDS + [
    "xgboost",
    "catboost",
    "neural_nets",
]
LARGE_DATA_REG_SEEDS = SMALL_DATA_REG_SEEDS + [
    "xgboost_regressor",
    "catboost_regressor",
    "neural_nets_regression",
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class HyperbandOptimizer:
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
        max_iter: int = 27,  # maximum iterations per configuration
        eta: int = 3,  # defines configuration downsampling rate (default = 3)
    ) -> None:
        self.name = name
        self.category = category

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

        self.candidate = {
            "score": -np.inf,
            "name": "nop",
            "params": {},
        }
        self.visited: Set[str] = set()
        self.predictions = Predictions(category=category)

        self.model_best_score = {}
        for seed in self.seeds:
            self.model_best_score[seed] = -np.inf

    def _hash_dict(self, name: str, dict_val: dict) -> str:
        return json.dumps(
            {"name": name, "val": dict_val}, sort_keys=True, cls=NpEncoder
        )

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

    def _sample(self, n: int) -> list:
        results = []
        for name in self.seeds:
            results.extend(self._sample_model(name, n))
        return results

    def _baseline(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        for seed in self.seeds:
            self._eval_params(seed, X, y, hyperparam_search_iterations=1)
        # TODO: balance methods

    def _eval_params(
        self, model_name: str, X: pd.DataFrame, y: pd.DataFrame, **params: Any
    ) -> float:
        model = self.predictions.get(model_name, **params)
        try:
            if self.category == "regression":
                out = evaluate_regression(model, X, y, n_folds=2)
                score = -out["clf"]["rmse"][0]
            else:
                out = evaluate_estimator(model, X, y, n_folds=2)
                score = out["clf"]["aucroc"][0]
        except BaseException as e:
            log.error(f"      >>> {self.name}:{model_name}: eval failed {e}")
            score = -9999

        if score > self.candidate["score"]:
            self.candidate = {
                "score": score,
                "params": params,
                "name": model_name,
            }
            if score > self.model_best_score[model_name]:
                self.model_best_score[model_name] = score

        return score

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        self._baseline(X, y)

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

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

        self.candidate["params"]["hyperparam_search_iterations"] = 100

        log.info(
            f"      >>> {self.name} -- best candidate {self.candidate['name']}: ({self.candidate['params']}) --- score : {self.candidate['score']}"
        )

        self.seeds = sorted(
            self.model_best_score, key=self.model_best_score.get, reverse=True  # type: ignore
        )[:2]

        return Predictions(category=self.category).get(
            self.candidate["name"], **self.candidate["params"]
        )


class BayesianOptimizer:
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
        patience: int = 10,
        inner_patience: int = 4,
    ):
        self.name = name
        self.category = category

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

        self.best_score = -9999
        self.best_candidate = self.seeds[0]
        self.best_params: dict = {}

    def evaluate_plugin(
        self,
        plugin_name: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        prev_best_score: float,
    ) -> tuple:
        # BO evaluation for a single plugin
        plugin = Predictions(category=self.category).get_type(plugin_name)

        early_stopping_patience: int = 0
        study, pruner = self.bo_studies[plugin_name]

        def evaluate_args(**kwargs: Any) -> float:
            model = plugin(**kwargs)
            try:
                if self.category == "regression":
                    out = evaluate_regression(model, X, y)
                    score = -out["clf"]["rmse"][0]
                else:
                    out = evaluate_estimator(model, X, y)
                    score = out["clf"]["aucroc"][0]
            except BaseException:
                score = -9999

            return score

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

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Any:
        best_score = -9999

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
        return Predictions(category=self.category).get(
            self.best_candidate, **self.best_params
        )


class SimpleOptimizer:
    def __init__(
        self,
        name: str,
        category: str,
        classifier_seed: list,
        regression_seed: list,
    ) -> None:
        self.name = name
        self.category = category

        if category == "classifier":
            self.seeds = classifier_seed
        else:
            self.seeds = regression_seed

        self.candidate = {
            "score": -np.inf,
            "name": "nop",
            "params": {},
        }
        self.predictions = Predictions(category=category)

        self.model_best_score = {}
        for seed in self.seeds:
            self.model_best_score[seed] = -np.inf

    def _eval_params(
        self, model_name: str, X: pd.DataFrame, y: pd.DataFrame, **params: Any
    ) -> float:
        model = self.predictions.get(model_name, **params)
        try:
            if self.category == "regression":
                out = evaluate_regression(model, X, y, n_folds=2)
                score = -out["clf"]["rmse"][0]
            else:
                out = evaluate_estimator(model, X, y, n_folds=2)
                score = out["clf"]["aucroc"][0]
        except BaseException as e:
            log.error(f"      >>> {self.name}:{model_name}: eval failed {e}")
            score = -9999

        if score > self.candidate["score"]:
            self.candidate = {
                "score": score,
                "params": params,
                "name": model_name,
            }
            if score > self.model_best_score[model_name]:
                self.model_best_score[model_name] = score

        return score

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        for seed in self.seeds:
            self._eval_params(seed, X, y)

        log.info(
            f"     >>> Column {self.name} <-- score {self.candidate['score']} <-- Model {self.candidate['name']}"
        )
        return Predictions(category=self.category).get(
            self.candidate["name"], **self.candidate["params"]
        )


class IterativeErrorCorrection:
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
        n_inner_iter: int = 50,
        n_outer_iter: int = 5,
    ):
        assert optimizer in [
            "hyperband",
            "bayesian",
            "simple",
        ], f"Invalid optimizer {optimizer}"

        self.study = study
        self.class_threshold = class_threshold
        self.baseline_imputer = Imputers().get(baseline_imputer)
        self.optimize_thresh = optimize_thresh
        self.optimize_thresh_upper = optimize_thresh_upper
        self.n_inner_iter = n_inner_iter
        self.n_outer_iter = n_outer_iter
        self.classifier_seed = classifier_seed
        self.regression_seed = regression_seed

        self.optimizer: Any
        if optimizer == "hyperband":
            self.optimizer = HyperbandOptimizer
        elif optimizer == "bayesian":
            self.optimizer = BayesianOptimizer
        else:
            self.optimizer = SimpleOptimizer

        self.avail_data_thresh = 500

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

    def _setup(self, X: pd.DataFrame) -> pd.DataFrame:
        # Encode the categorical columns
        # Reset internal caches
        X = pd.DataFrame(X).copy()

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
                    **self._select_seeds(avail_cnt),
                )
            else:
                self.continuous_cols.append(col)
                self.column_to_optimizer[col] = optimizer(
                    col,
                    "regression",
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

    def _tear_down(self, X: pd.DataFrame) -> pd.DataFrame:
        # Revert the encoding after processing the data
        for col in self.encoders:
            X[col] = self.encoders[col].inverse_transform(X[col].astype(int))

        return X

    def _get_neighbors_for_col(self, col: str) -> list:
        covs = list(self.all_cols)
        covs.remove(col)

        return covs

    def _optimize_model_for_column(self, X: pd.DataFrame, col: str) -> dict:
        # BO evaluation for a single column
        if self.mask[col].sum() == 0:
            return self.column_to_model

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

        candidate = self.column_to_optimizer[col].evaluate(X_train, y_train)

        self.column_to_model[col] = candidate

        return self.column_to_model

    def _optimize(self, X: pd.DataFrame) -> dict:
        # BO evaluation to select the best models for each columns

        for col in self.imputation_order:
            self._optimize_model_for_column(X, col)

        return self.column_to_model

    def _impute_single_column(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
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

        est = copy.deepcopy(self.column_to_model[col])

        est.fit(X_train, y_train)

        X[col][self.mask[col]] = est.predict(covs[self.mask[col]]).values.squeeze()

        col_min, col_max = self.limits[col]
        X[col][self.mask[col]] = np.clip(X[col][self.mask[col]], col_min, col_max)

        return X

    def _iterate_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        # Run an iteration of imputation on all columns
        random.shuffle(self.imputation_order)
        for col in self.imputation_order:
            X = self._impute_single_column(X, col)
        return X

    def _initial_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        # Use baseline imputer for initial values
        return self.baseline_imputer.fit_transform(X)

    def _is_categorical(self, X: pd.DataFrame, col: str) -> bool:
        # Helper for filtering categorical columns
        return len(X[col].unique()) < self.class_threshold

    def _missing_indicator(self, X: pd.DataFrame) -> pd.DataFrame:
        # Helper for generating missingness mask
        return pd.DataFrame(
            MissingIndicator(features="all").fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame) -> "IterativeErrorCorrection":
        # Run imputation
        X = self._setup(X)

        Xt_init = self._initial_imputation(X)
        Xt_init.columns = X.columns

        Xt = Xt_init.copy()

        for out_it in range(self.n_outer_iter):
            log.info(f"  > BO iter {out_it}")

            self._optimize(Xt.copy())

            for it in range(self.n_inner_iter):
                Xt_prev = Xt.copy()
                Xt = self._iterate_imputation(Xt_prev.copy())

                inf_norm = np.linalg.norm(Xt - Xt_prev, ord=np.inf, axis=None)
                if inf_norm < TOL:
                    log.info(
                        f"     >>>> Early stopping on iteration diff {mean_squared_error(Xt_prev, Xt)}"
                    )
                    break

        return self._tear_down(Xt)


class HyperImputePlugin(base.ImputerPlugin):
    """HyperImpute strategy."""

    def __init__(
        self,
        classifier_seed: list = LARGE_DATA_CLF_SEEDS,
        regression_seed: list = LARGE_DATA_REG_SEEDS,
        optimizer: str = "simple",
        baseline_imputer: str = "mean",
        class_threshold: int = 20,
        optimize_thresh: int = 1000,
        n_inner_iter: int = 50,
        n_outer_iter: int = 5,
    ) -> None:
        super().__init__()

        self.model = IterativeErrorCorrection(
            "hyperimpute_plugin",
            classifier_seed=classifier_seed,
            regression_seed=regression_seed,
            optimizer=optimizer,
            baseline_imputer=baseline_imputer,
            class_threshold=class_threshold,
            optimize_thresh=optimize_thresh,
            n_inner_iter=n_inner_iter,
            n_outer_iter=n_outer_iter,
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


plugin = HyperImputePlugin
