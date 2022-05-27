# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple

# third party
import numpy as np
from optuna.trial import Trial
from pydantic import validate_arguments


class Params(metaclass=ABCMeta):
    @validate_arguments
    def __init__(self, name: str, bounds: Tuple[Any, Any]) -> None:
        self.name = name
        self.bounds = bounds

    @abstractmethod
    def get(self) -> List[Any]:
        ...

    @abstractmethod
    def sample(self, trial: Trial) -> Any:
        ...

    @abstractmethod
    def sample_np(self) -> Any:
        ...


class Categorical(Params):
    @validate_arguments
    def __init__(self, name: str, choices: List[Any]) -> None:
        super().__init__(name, (min(choices), max(choices)))
        self.name = name
        self.choices = choices

    def get(self) -> List[Any]:
        return [self.name, self.choices]

    def sample(self, trial: Trial) -> Any:
        return trial.suggest_categorical(self.name, self.choices)

    def sample_np(self) -> Any:
        return np.random.choice(self.choices, 1)[0]


class Float(Params):
    @validate_arguments
    def __init__(self, name: str, low: float, high: float) -> None:
        low = float(low)
        high = float(high)

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high)

    def sample_np(self) -> Any:
        return np.random.uniform(self.low, self.high)


class Integer(Params):
    @validate_arguments
    def __init__(self, name: str, low: int, high: int, step: int = 1) -> None:
        self.low = low
        self.high = high
        self.step = step

        super().__init__(name, (low, high))
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.choices = [val for val in range(low, high + 1, step)]

    def get(self) -> List[Any]:
        return [self.name, self.low, self.high, self.step]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample(self, trial: Trial) -> Any:
        return trial.suggest_int(self.name, self.low, self.high, self.step)

    def sample_np(self) -> Any:
        return np.random.choice(self.choices, 1)[0]
