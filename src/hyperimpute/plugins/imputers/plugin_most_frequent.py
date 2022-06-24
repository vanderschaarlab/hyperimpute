# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.impute import SimpleImputer

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base


class MostFrequentPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Most Frequent Imputation strategy.

    Method:
        The Most Frequent Imputation strategy replaces the missing using the most frequent value along each column.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("most_frequent")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    """

    def __init__(self, random_state: int = 0) -> None:
        super().__init__(random_state=random_state)

        self._model = SimpleImputer(strategy="most_frequent")

    @staticmethod
    def name() -> str:
        return "most_frequent"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MostFrequentPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)


plugin = MostFrequentPlugin
