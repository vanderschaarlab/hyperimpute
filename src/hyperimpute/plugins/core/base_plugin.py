# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List

# third party
import numpy as np
from optuna.trial import Trial
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.utils.cast as cast

# hyperimpute relative
from .params import Params


class Plugin(metaclass=ABCMeta):
    """Base class for all plugins.
    Each derived class must implement the following methods:
        type() - a static method that returns the type of the plugin. e.g., imputation, preprocessing, prediction, etc.
        subtype() - optional method that returns the subtype of the plugin. e.g. Potential subtypes:
            - preprocessing: feature_scaling, dimensionality reduction
            - prediction: classifiers, prediction, survival analysis
        name() - a static method that returns the name of the plugin. e.g., EM, mice, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.
        _fit() - internal method, called by `fit` on each training set.
        _transform() - internal method, called by `transform`. Used by imputation or preprocessing plugins.
        _predict() - internal method, called by `predict`. Used by classification/prediction plugins.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self) -> None:
        self.output = pd.DataFrame

    def change_output(self, output: str) -> None:
        assert output in ["pandas", "numpy"], "Invalid output type"
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

    @staticmethod
    @abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        ...

    @classmethod
    def sample_hyperparameters(
        cls, trial: Trial, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    @classmethod
    def sample_hyperparameters_np(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample_np()

        return results

    @classmethod
    def hyperparameter_space_fqdn(cls, *args: Any, **kwargs: Any) -> List[Params]:
        res = []
        for param in cls.hyperparameter_space(*args, **kwargs):
            fqdn_param = param
            fqdn_param.name = (
                cls.type() + "." + cls.subtype() + "." + cls.name() + "." + param.name
            )
            res.append(fqdn_param)

        return res

    @classmethod
    def sample_hyperparameters_fqdn(
        cls, trial: Trial, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        param_space = cls.hyperparameter_space_fqdn(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        ...

    @staticmethod
    @abstractmethod
    def subtype() -> str:
        ...

    @classmethod
    def fqdn(cls) -> str:
        return cls.type() + "." + cls.subtype() + "." + cls.name()

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(self.fit(X, *args, *kwargs).transform(X))

    def fit_predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(self.fit(X, *args, *kwargs).predict(X))

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        X = cast.to_dataframe(X)
        return self._fit(X, *args, **kwargs)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = cast.to_dataframe(X)
        return self.output(self._transform(X))

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = cast.to_dataframe(X)
        return self.output(self._predict(X, *args, *kwargs))

    @abstractmethod
    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    @abstractmethod
    def save(self) -> bytes:
        ...

    @classmethod
    @abstractmethod
    def load(cls, buff: bytes) -> "Plugin":
        ...
