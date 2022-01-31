# stdlib
import glob
from os.path import basename, dirname, isfile, join

# hyperimpute absolute
from hyperimpute.plugins.core.base_plugin import PluginLoader

# hyperimpute relative
from .base import ImputerPlugin  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Imputers(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, ImputerPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Imputers",
    "ImputerPlugin",
]
