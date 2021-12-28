# stdlib
from typing import Any, Dict, List, Tuple, Type, Union

# hyperimpute absolute
from hyperimpute.plugins.explainers import Explainers  # noqa: F401,E402
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.plugins.prediction import Predictions
from hyperimpute.plugins.preprocessors import Preprocessors
import hyperimpute.plugins.utils  # noqa: F401,E402

# hyperimpute relative
from .core import base_plugin  # noqa: F401,E402


class Plugins:
    def __init__(self) -> None:
        self._plugins: Dict[
            str, Dict[str, Union[Imputers, Predictions, Preprocessors]]
        ] = {
            "imputer": {
                "default": Imputers(),
            },
            "prediction": {
                "classifier": Predictions(category="classifier"),
                "risk_estimation": Predictions(category="risk_estimation"),
            },
            "preprocessor": {
                "feature_scaling": Preprocessors(category="feature_scaling"),
                "dimensionality_reduction": Preprocessors(
                    category="dimensionality_reduction"
                ),
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
        assert "." in fqdn
        cat, subtype, name = fqdn.split(".")

        res.append(plugins.get_type(cat, subtype, name))

    return tuple(res)
