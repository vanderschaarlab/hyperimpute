# stdlib
import time
from typing import Any, List, Union

# third party
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
from sklearn.impute import IterativeImputer

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
from hyperimpute.plugins.prediction import Predictions
import hyperimpute.plugins.utils.decorators as decorators
import hyperimpute.utils.serialization as serialization


class IterativeImputerPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the IterativeImputer strategy.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("missforest")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
             0    1    2    3
        0  1.0  1.0  1.0  1.0
        1  1.0  1.9  1.9  1.0
        2  1.0  2.0  2.0  1.0
        3  2.0  2.0  2.0  2.0
    """

    def __init__(
        self,
        regressor: str = "xgboost_regressor",
        max_iter: int = 100,
        random_state: Union[int, None] = 0,
        model: Any = None,
    ) -> None:
        super().__init__()

        if model:
            self._model = model
            return

        if not random_state:
            random_state = int(time.time())

        self._model = IterativeImputer(
            estimator=Predictions(category="regression").get(regressor).model,  # type: ignore
            random_state=random_state,
            max_iter=max_iter,
        )

    @staticmethod
    def name() -> str:
        return "iterative_imputer"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_iter", 10, 300, 10),
        ]

    @decorators.benchmark
    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "IterativeImputerPlugin":
        self._model.fit(X, *args, **kwargs)

        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self._model)

    @classmethod
    def load(cls, buff: bytes) -> "IterativeImputerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = IterativeImputerPlugin
