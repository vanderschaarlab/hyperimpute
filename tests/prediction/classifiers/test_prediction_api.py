# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.prediction import Predictions
from hyperimpute.plugins.prediction.classifiers import ClassifierPlugin


@pytest.fixture
def ctx() -> Predictions:
    return Predictions()


class Mock(ClassifierPlugin):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "test"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Mock":
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return {}

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return {}

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "Mock":
        return cls()


class Invalid:
    def __init__(self) -> None:
        pass


def test_load(ctx: Predictions) -> None:
    assert len(ctx._plugins) > 0
    assert len(ctx._plugins) == len(ctx)


def test_list(ctx: Predictions) -> None:
    assert "catboost" in ctx.list()


def test_add_get(ctx: Predictions) -> None:
    ctx.add("mock", Mock)

    assert "mock" in ctx.list()

    mock = ctx.get("mock")

    assert mock.name() == "test"

    ctx.reload()
    assert "mock" not in ctx.list()


def test_add_get_invalid(ctx: Predictions) -> None:
    with pytest.raises(ValueError):
        ctx.add("invalid", Invalid)

    assert "mock" not in ctx.list()

    with pytest.raises(ValueError):
        ctx.get("mock")


def test_iter(ctx: Predictions) -> None:
    for v in ctx:
        assert ctx[v].name() != ""
