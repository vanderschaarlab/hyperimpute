# stdlib
import glob
from os.path import basename, dirname, isfile, join

# hyperimpute absolute
from hyperimpute.plugins.core.base_plugin import PluginLoader
from hyperimpute.plugins.prediction.regression.base import (  # noqa: F401,E402
    RegressionPlugin,
)

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Regression(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, RegressionPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Regression",
    "RegressionPlugin",
]
