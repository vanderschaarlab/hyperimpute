# Imputation Plugins

### :information_source: About


Dataset imputation is the process of replacing missing data with substituted values.


This module provides the default imputation plugins supported by HyperImpute.The library automatically loads every file that follows the pattern `*_plugin.py` and exports a class derived from the [`ImputerPlugin`](base.py) interface.

### :zap: Plugins
The following table contains the default imputation plugins:

| Strategy | Description| Based on| Code | Tests|
|--- | --- | --- | --- | --- |
|**Mean**|Replace the missing values using the mean along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)| [`plugin_mean.py`](plugin_mean.py) | [`test_mean.py`](../../../../tests/plugins/imputers/test_mean.py) |
|**Median**|Replace the missing values using the median along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)| [`plugin_median.py`](plugin_median.py) | [`test_median.py`](../../../../tests/plugins/imputers/test_median.py)|
|**Most-frequent**|Replace the missing values using the most frequent value along each column|[`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)|[`plugin_most_freq.py`](plugin_most_freq.py) | [`test_most_freq.py`](../../../../tests/plugins/imputers/test_most_freq.py) |
|**MissForest**|Iterative imputation method based on Random Forests| [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)| [`plugin_missforest.py`](plugin_missforest.py) |[`test_missforest.py`](../../../../tests/plugins/imputers/test_missforest.py) |
|**ICE**| Iterative imputation method based on regularized linear regression | [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_ice.py`](plugin_ice.py)| [`test_ice.py`](../../../../tests/plugins/imputers/test_ice.py)|
|**MICE**| Multiple imputations based on ICE| [`IterativeImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer) and [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)| [`plugin_mice.py`](plugin_mice.py) |[`test_mice.py`](../../../../tests/plugins/imputers/test_mice.py) |
|**SoftImpute**|Low-rank matrix approximation via nuclear-norm regularization| [`Original paper`](https://jmlr.org/papers/volume16/hastie15a/hastie15a.pdf)| [`plugin_softimpute.py`](plugin_softimpute.py)|[`test_softimpute.py`](../../../../tests/plugins/imputers/test_softimpute.py) |
|**EM**|Iterative procedure which uses other variables to impute a value (Expectation), then checks whether that is the value most likely (Maximization)|[`EM imputation algorithm`](https://joon3216.github.io/research_materials/2019/em_imputation.html)|[`plugin_em.py`](plugin_em.py) |[`test_em.py`](../../../../tests/plugins/imputers/test_em.py) |
|**Sinkhorn**|Based on the Optimal transport distances between random batches|[`Original paper`](https://arxiv.org/pdf/2002.03860.pdf)|[`plugin_sinkhorn.py`](plugin_sinkhorn.py) | [`test_sinkhorn.py`](../../../../tests/plugins/imputers/test_sinkhorn.py)|

### :hammer: Writing a new imputation plugin
Every **HyperImpute plugin** must implement the **`Plugin`** interface provided by [`hyperimpute/plugins/core/base_plugin.py`](../core/base_plugin.py).

Each **HyperImpute imputation plugin** must implement the **`ImputerPlugin`** interface provided by [`hyperimpute/plugins/imputers/base.py`](base.py)

:heavy_exclamation_mark: __Warning__ : If a plugin doesn't override all the abstract methods, it won't be loaded by the library.



Every imputation plugin **must implement** the following methods:
- *name()* - a static method that returns the name of the plugin. e.g., EM, mice, etc.

- *hyperparameter_space()* - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `params.Params` derived objects.

- *_fit()* - internal implementation, called by the `fit` method.
- *_transform()* - internal implementation, called by the `transform` method.

### :cyclone: Example: Adding a new plugin

```
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
