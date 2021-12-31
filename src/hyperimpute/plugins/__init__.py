# stdlib
from typing import Any, Dict, List, Tuple, Type, Union

# hyperimpute absolute
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.prediction import Predictions
import hyperimpute.plugins.utils  # noqa: F401,E402

# hyperimpute relative
from .core import base_plugin  # noqa: F401,E402


class Plugins:
    def __init__(self) -> None:
        self._plugins: Dict[str, Dict[str, Union[Imputers, Predictions]]] = {
            "imputer": {
                "default": Imputers(),
            },
            "prediction": {
                "classifier": Predictions(category="classifier"),
                "regression": Predictions(category="regression"),
            },
        }

    def list(self) -> dict:
        res: Dict[str, Dict[str, List[str]]] = {}
        for src in self._plugins:
            res[src] = {}
            for subtype in self._plugins[src]:
                res[src][subtype] = self._plugins[src][subtype].list()
        return res

    def add(self, cat: str, subtype: str, name: str, cls: Type) -> "Plugins":
        self._plugins[cat][subtype].add(name, cls)

        return self

    def get(
        self, cat: str, name: str, subtype: str, *args: Any, **kwargs: Any
    ) -> base_plugin.Plugin:
        return self._plugins[cat][subtype].get(name, *args, **kwargs)

    def get_type(self, cat: str, subtype: str, name: str) -> Type:
        return self._plugins[cat][subtype].get_type(name)


def group(names: List[str]) -> Tuple[Type, ...]:
    res = []

    plugins = Plugins()
    for fqdn in names:
        if "." not in fqdn:
            raise RuntimeError(f"invalid fqdn {fqdn}")

        cat, subtype, name = fqdn.split(".")

        res.append(plugins.get_type(cat, subtype, name))

    return tuple(res)
