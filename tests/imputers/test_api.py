# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest

# hyperimpute absolute
from hyperimpute.plugins.imputers import ImputerPlugin, Imputers


@pytest.fixture
def ctx() -> Imputers:
    return Imputers()


class Mock(ImputerPlugin):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "test"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Any]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Mock":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return {}

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "Mock":
        return cls()


class Invalid:
    def __init__(self) -> None:
        pass


def test_load(ctx: Imputers) -> None:
    assert len(ctx._plugins) == 0
    ctx.get("mean")
    assert len(ctx._plugins) == 1


def test_list(ctx: Imputers) -> None:
    ctx.get("mean")
    assert "mean" in ctx.list()


def test_add_get(ctx: Imputers) -> None:
    ctx.add("mock", Mock)

    assert "mock" in ctx.list()

    mock = ctx.get("mock")

    assert mock.name() == "test"

    ctx.reload()
    assert "mock" not in ctx.list()


def test_add_get_invalid(ctx: Imputers) -> None:
    with pytest.raises(ValueError):
        ctx.add("invalid", Invalid)

    assert "mock" not in ctx.list()

    with pytest.raises(ValueError):
        ctx.get("mock")


def test_iter(ctx: Imputers) -> None:
    for v in ctx:
        assert ctx[v].name() != ""
