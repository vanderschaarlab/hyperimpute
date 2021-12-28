# stdlib
import copy
from typing import Any
import warnings

# third party
from IPython.display import HTML, display
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tabulate

# hyperimpute absolute
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.prediction import Predictions
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.distributions import enable_reproducible_results

warnings.filterwarnings("ignore")
enable_reproducible_results()


imputers = Imputers()

# Simulation


def ampute(
    x: pd.DataFrame, mechanism: str, p_miss: float, column_limit: int = 8
) -> tuple:
    columns = x.columns
    column_limit = min(len(columns), column_limit)

    sampled_columns = columns[
        np.random.choice(len(columns), size=column_limit, replace=False)
    ]
    x_simulated = simulate_nan(x[sampled_columns].values, p_miss, mechanism)

    isolated_mask = pd.DataFrame(x_simulated["mask"], columns=sampled_columns)
    isolated_x_miss = pd.DataFrame(x_simulated["X_incomp"], columns=sampled_columns)

    mask = pd.DataFrame(np.zeros(x.shape), columns=columns)
    mask[sampled_columns] = pd.DataFrame(isolated_mask, columns=sampled_columns)

    x_miss = pd.DataFrame(x.copy(), columns=columns)
    x_miss[sampled_columns] = isolated_x_miss

    return (
        pd.DataFrame(x, columns=columns),
        x_miss,
        mask,
    )


def scale_data(X: pd.DataFrame) -> pd.DataFrame:
    preproc = MinMaxScaler()
    cols = X.columns
    return pd.DataFrame(preproc.fit_transform(X), columns=cols)


def simulate_scenarios(X: pd.DataFrame) -> pd.DataFrame:
    X = scale_data(X)

    datasets: dict = {}

    mechanisms = ["MAR", "MNAR", "MCAR"]
    percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

    for ampute_mechanism in mechanisms:
        for p_miss in percentages:
            if ampute_mechanism not in datasets:
                datasets[ampute_mechanism] = {}

            datasets[ampute_mechanism][p_miss] = ampute(X, ampute_mechanism, p_miss)

    return datasets


def ws_score(imputed: pd.DataFrame, ground: pd.DataFrame) -> pd.DataFrame:
    res = 0
    for col in range(ground.shape[1]):
        res += wasserstein_distance(
            np.asarray(ground)[:, col], np.asarray(imputed)[:, col]
        )
    return res


def benchmark_using_downstream_model(
    X: pd.DataFrame, imputed_X: pd.DataFrame, y: pd.DataFrame
) -> float:
    outcomes = np.unique(y)
    if len(outcomes) < 10:
        ground_model = Predictions(category="classifier").get("xgboost")
        imputed_model = Predictions(category="classifier").get("xgboost")
    else:
        ground_model = Predictions(category="regression").get("xgboost_regressor")
        imputed_model = Predictions(category="regression").get("xgboost_regressor")

    (
        X_train,
        X_test,
        imputed_X_train,
        imputed_X_test,
        y_train,
        y_test,
    ) = train_test_split(X, imputed_X, y)

    ground_model.fit(X_train, y_train)
    ground_y_hat = ground_model.predict(X_test)
    ground_err = mean_squared_error(ground_y_hat, y_test)

    imputed_model.fit(imputed_X_train, y_train)
    imputed_y_hat = imputed_model.predict(X_test)
    imputed_err = mean_squared_error(imputed_y_hat, y_test)

    return imputed_err - ground_err


def benchmark_model(
    name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    X_miss: np.ndarray,
    mask: np.ndarray,
) -> tuple:
    imputed = model.fit_transform(X_miss.copy())

    downstream_score = benchmark_using_downstream_model(X, imputed, y)
    distribution_score = ws_score(imputed, X)
    rmse_score = RMSE(np.asarray(imputed), np.asarray(X), np.asarray(mask))

    return rmse_score, distribution_score, downstream_score


def benchmark_standard(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    X_miss: np.ndarray,
    mask: np.ndarray,
) -> tuple:
    imputer = imputers.get(model_name)
    return benchmark_model(model_name, imputer, X, y, X_miss, mask)


def evaluate_dataset(
    name: str,
    evaluated_model: Any,
    X_raw: pd.DataFrame,
    y: pd.DataFrame,
    ref_methods: list = ["mean", "missforest", "ice", "gain", "sinkhorn", "softimpute"],
    scenarios: list = ["MAR", "MCAR", "MNAR"],
    miss_pct: list = [0.1, 0.3, 0.5],
    debug: bool = False,
) -> tuple:
    imputation_scenarios = simulate_scenarios(X_raw)

    rmse_results: dict = {}
    distr_results: dict = {}
    downstream_results: dict = {}

    for scenario in scenarios:

        rmse_results[scenario] = {}
        distr_results[scenario] = {}
        downstream_results[scenario] = {}

        for missingness in miss_pct:
            if debug:
                print("  > eval ", scenario, missingness)
            rmse_results[scenario][missingness] = {}
            distr_results[scenario][missingness] = {}
            downstream_results[scenario][missingness] = {}

            try:
                x, x_miss, mask = imputation_scenarios[scenario][missingness]

                (
                    our_rmse_score,
                    our_distribution_score,
                    our_downstream_score,
                ) = benchmark_model(
                    name, copy.deepcopy(evaluated_model), x, y, x_miss, mask
                )
                rmse_results[scenario][missingness]["our"] = our_rmse_score
                distr_results[scenario][missingness]["our"] = our_distribution_score
                downstream_results[scenario][missingness]["our"] = our_downstream_score

                for method in ref_methods:
                    x, x_miss, mask = imputation_scenarios[scenario][missingness]

                    (
                        mse_score,
                        distribution_score,
                        downstream_score,
                    ) = benchmark_standard(method, x, y, x_miss, mask)
                    rmse_results[scenario][missingness][method] = mse_score
                    distr_results[scenario][missingness][method] = distribution_score
                    downstream_results[scenario][missingness][method] = downstream_score
            except BaseException as e:
                print("scenario failed", str(e))
                continue
    return rmse_results, distr_results, downstream_results


def evaluate_dataset_repeated_internal(
    name: str,
    evaluated_model: Any,
    X_raw: pd.DataFrame,
    y: pd.DataFrame,
    ref_methods: list = ["mean", "missforest", "ice", "gain", "sinkhorn", "softimpute"],
    scenarios: list = ["MNAR"],
    miss_pct: list = [0.1, 0.3, 0.5, 0.7],
    n_iter: int = 2,
    debug: bool = False,
) -> None:
    def add_metrics(
        store: dict, scenario: str, missingness: float, method: str, score: float
    ) -> None:
        if scenario not in store:
            store[scenario] = {}
        if missingness not in store[scenario]:
            store[scenario][missingness] = {}
        if method not in store[scenario][missingness]:
            store[scenario][missingness][method] = []

        store[scenario][missingness][method].append(score)

    rmse_results_dict: dict = {}
    distr_results_dict: dict = {}
    downstream_results_dict: dict = {}

    for it in range(n_iter):
        if debug:
            print("> evaluation trial ", it)
        (
            local_rmse_results,
            local_distr_results,
            local_downstream_results,
        ) = evaluate_dataset(
            name=name,
            evaluated_model=evaluated_model,
            X_raw=X_raw,
            y=y,
            ref_methods=ref_methods,
            scenarios=scenarios,
            debug=debug,
            miss_pct=miss_pct,
        )
        for scenario in local_rmse_results:
            for missingness in local_rmse_results[scenario]:
                for method in local_rmse_results[scenario][missingness]:
                    add_metrics(
                        rmse_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_rmse_results[scenario][missingness][method],
                    )
                    add_metrics(
                        distr_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_distr_results[scenario][missingness][method],
                    )
                    add_metrics(
                        downstream_results_dict,
                        scenario,
                        missingness,
                        method,
                        local_downstream_results[scenario][missingness][method],
                    )

    rmse_results = []
    distr_results = []
    downstream_results = []

    for scenario in rmse_results_dict:

        for missingness in rmse_results_dict[scenario]:

            local_rmse_results = [scenario, missingness]
            local_distr_results = [scenario, missingness]
            local_downstream_results = [scenario, missingness]

            for method in ["our"] + ref_methods:
                local_rmse_results.append(
                    np.mean(rmse_results_dict[scenario][missingness][method])
                )
                local_distr_results.append(
                    np.mean(distr_results_dict[scenario][missingness][method])
                )
                local_downstream_results.append(
                    np.mean(downstream_results_dict[scenario][missingness][method])
                )

            rmse_results.append(local_rmse_results)
            distr_results.append(local_distr_results)
            downstream_results.append(local_downstream_results)

    headers = ["Scenario", "miss_pct [0, 1]"] + ["Our method"] + ref_methods

    sep = "\n==========================================================\n\n"
    print("RMSE score")
    display(HTML(tabulate.tabulate(rmse_results, headers=headers, tablefmt="html")))

    print(sep + "Wasserstein score")

    display(HTML(tabulate.tabulate(distr_results, headers=headers, tablefmt="html")))

    print(sep + "Downstream model prediction error")
    display(
        HTML(tabulate.tabulate(downstream_results, headers=headers, tablefmt="html"))
    )
