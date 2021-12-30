# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any


class Hooks(metaclass=ABCMeta):
    @abstractmethod
    def cancel(self) -> bool:
        ...

    @abstractmethod
    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        ...
