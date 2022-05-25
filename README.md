<h2 align="center">
  hyperimpute
</h2>

<h4 align="center">
    A library for NaNs and nulls
</h4>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![hyperimpute Tests](https://github.com/vanderschaarlab/hyperimpute/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/hyperimpute/actions/workflows/test.yml)
[![CodeQL](https://github.com/vanderschaarlab/hyperimpute/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/vanderschaarlab/hyperimpute/actions/workflows/codeql-analysis.yml)

<div align="center">

</div>

Dataset imputation is the process of replacing missing data with substituted values.

hyperimpute features:
- :key: New iterative imputation method: HyperImpute.
- :cyclone: Classic methods like MICE, MissForest, GAIN etc.
- :fire: Pluginable architecture.

## :rocket: Installation

The library can be installed using
```bash
$ pip install .
```

## :boom: Sample Usage
List available imputers
```python
from hyperimpute.plugins.imputers import Imputers

imputers = Imputers()

imputers.list()
```
Impute a dataset using one of the available methods
```python
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])

plugin = Imputers().get("miracle")
out = plugin.fit_transform(X.copy())
print(method, out)
```
Specify the baseline models for HyperImpute
```python
import pandas as pd
import numpy as np
from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])

plugin = Imputers().get(
    "hyperimpute",
    optimizer="hyperband",
    classifier_seed=["logistic_regression"],
    regression_seed=["linear_regression"],
)

out = plugin.fit_transform(X.copy())
print(out)
```
Use an imputer with a SKLearn pipeline
```python
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from hyperimpute.plugins.imputers import Imputers

X = pd.DataFrame([[1, 1, 1, 1], [4, 5, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]])
y = pd.Series([1, 2, 1, 2])

imputer = Imputers().get("hyperimpute")

estimator = Pipeline(
    [
        ("imputer", imputer),
        ("forest", RandomForestRegressor(random_state=0, n_estimators=100)),
    ]
)

estimator.fit(X, y)
```
Write a new imputation plugin
```python
from sklearn.impute import KNNImputer
from hyperimpute.plugins.imputers import Imputers, ImputerPlugin

imputers = Imputers()

knn_imputer = "custom_knn"

class KNN(ImputerPlugin):
    def __init__(self) -> None:
        super().__init__()
        self._model = KNNImputer(n_neighbors=2, weights="uniform")

    @staticmethod
    def name():
        return knn_imputer

    @staticmethod
    def hyperparameter_space():
        return []

    def _fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)
        return self

    def _transform(self, *args, **kwargs):
        return self._model.transform(*args, **kwargs)

imputers.add(knn_imputer, KNN)

assert imputers.get(knn_imputer) is not None
```
Benchmark imputation models on a dataset
```python
from sklearn.datasets import load_iris
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.utils.benchmarks import compare_models

X, y = load_iris(as_frame=True, return_X_y=True)

imputer = Imputers().get("hyperimpute")

compare_models(
    name="example",
    evaluated_model=imputer,
    X_raw=X,
    ref_methods=["ice", "missforest"],
    scenarios=["MAR"],
    miss_pct=[0.1, 0.3],
    n_iter=2,
)
```

## 📓 Tutorials
 - [Tutorial 0: Imputation basics](tutorials/tutorial_00_imputer_plugins.ipynb)
 - [Tutorial 1: AutoML for imputation](tutorials/tutorial_01_bayesian_optimization_over_imputers.ipynb)
 - [Tutorial 2: HyperImpute example](tutorials/experiments_01_hyperimpute_with_naive_search.ipynb)
 - [Tutorial 3: HyperImpute with Hyperband](tutorials/experiments_01_hyperimpute_with_hyperband.ipynb)

## :zap: Imputation methods
The following table contains the default imputation plugins:

| Strategy | Description| Code |
|--- | --- | --- |
|**HyperImpute**|Iterative imputer using both regression and classification methods based on linear models, trees, XGBoost, CatBoost and neural nets| [`plugin_hyperimpute.py`](src/hyperimpute/plugins/imputers/plugin_hyperimpute.py) |
|**Mean**|Replace the missing values using the mean along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)| [`plugin_mean.py`](src/hyperimpute/plugins/imputers/plugin_mean.py) |
|**Median**|Replace the missing values using the median along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) |  [`plugin_median.py`](src/hyperimpute/plugins/imputers/plugin_median.py) |
|**Most-frequent**|Replace the missing values using the most frequent value along each column with [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)|[`plugin_most_freq.py`](src/hyperimpute/plugins/imputers/plugin_most_freq.py) |
|**MissForest**|Iterative imputation method based on Random Forests using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)| [`plugin_missforest.py`](src/hyperimpute/plugins/imputers/plugin_missforest.py) |
|**ICE**| Iterative imputation method based on regularized linear regression using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_ice.py`](src/hyperimpute/plugins/imputers/plugin_ice.py)|
|**MICE**| Multiple imputations based on ICE using [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_mice.py`](src/hyperimpute/plugins/imputers/plugin_mice.py) |
|**SoftImpute**|  [`Low-rank matrix approximation via nuclear-norm regularization`](https://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf)| [`plugin_softimpute.py`](src/hyperimpute/plugins/imputers/plugin_softimpute.py)|
|**EM**|Iterative procedure which uses other variables to impute a value (Expectation), then checks whether that is the value most likely (Maximization) - [`EM imputation algorithm`](https://joon3216.github.io/research_materials/2019/em_imputation.html)|[`plugin_em.py`](src/hyperimpute/plugins//imputers/plugin_em.py) |
|**Sinkhorn**|[`Missing Data Imputation using Optimal Transport`](https://arxiv.org/pdf/2002.03860.pdf)|[`plugin_sinkhorn.py`](src/hyperimpute/plugins/imputers/plugin_sinkhorn.py) |
|**GAIN**|[`GAIN: Missing Data Imputation using Generative Adversarial Nets`](https://arxiv.org/abs/1806.02920)|[`plugin_gain.py`](src/hyperimpute/plugins/imputers/plugin_gain.py) |
|**MIRACLE**|[`MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms`](https://arxiv.org/abs/2111.03187)|[`plugin_miracle.py`](src/hyperimpute/plugins/imputers/plugin_miracle.py) |
|**MIWAE**|[`MIWAE: Deep Generative Modelling and Imputation of Incomplete Data`](https://arxiv.org/abs/1812.02633)|[`plugin_miwae.py`](src/hyperimpute/plugins/imputers/plugin_miwae.py) |


## :hammer: Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```
