# stdlib
from importlib.abc import Loader
import glob
import importlib.util
from os.path import basename, dirname, isfile, join
from typing import Any, Dict, Generator, List, Type

# hyperimpute absolute
import hyperimpute.logger as log

# hyperimpute relative
from .base import ImputerPlugin  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Imputers:
    def __init__(self) -> None:
        self._plugins: Dict[str, Type] = {}

        self._load_default_plugins()

    def _load_default_plugins(self) -> None:

        for plugin in plugins:
            name = basename(plugin)
            spec = importlib.util.spec_from_file_location(name, plugin)
            assert isinstance(spec.loader, Loader)

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            try:
                cls = mod.plugin  # type: ignore
            except BaseException as e:
                log.critical(f"module {name} load failed {e}")
                continue

            log.debug(f"Loaded plugin {cls.type()} - {cls.name()}")
            self.add(cls.name(), cls)

    def list(self) -> List[str]:
        return list(self._plugins.keys())

    def types(self) -> List[Type]:
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "Imputers":
        if name in self._plugins:
            raise ValueError(f"Plugin {name} already exists.")

        if not issubclass(cls, ImputerPlugin):
            raise ValueError(f"Plugin {name} must derive the ImputerPlugin interface.")

        self._plugins[name] = cls

        return self

    def get(self, name: str, **kwargs: Any) -> ImputerPlugin:
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name](**kwargs)

    def get_type(self, name: str) -> Type:
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        return len(self.list())

    def __getitem__(self, key: str) -> ImputerPlugin:
        return self.get(key)

    def reload(self) -> "Imputers":
        self._plugins = {}
        self._load_default_plugins()
        return self


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Imputers",
    "ImputerPlugin",
]
