# stdlib
from typing import Any, List

# third party
import pandas as pd

# hyperimpute absolute
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base


class NopPlugin(base.ImputerPlugin):
    """Imputer plugin that doesn't alter the dataset."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "nop"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "NopPlugin":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


plugin = NopPlugin
