# stdlib
import copy
from importlib.abc import Loader
import importlib.util
import os
from pathlib import Path
from typing import Any, Optional, Union

# third party
import cloudpickle
from pydantic import validate_arguments

# hyperimpute absolute
from hyperimpute.version import MAJOR_VERSION

module_path = Path(__file__).resolve()
module_parent_path = module_path.parent


class Serializable:
    """Utility class for model persistence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        derived_module_path: Optional[Path] = None

        search_module = self.__class__.__module__
        if not search_module.endswith(".py"):
            search_module = search_module.split(".")[-1]
            search_module += ".py"

        for path in module_path.parent.parent.rglob(search_module):
            derived_module_path = path
            break

        self.module_relative_path: Optional[Path] = None

        if derived_module_path is not None:
            relative_path = Path(
                os.path.relpath(derived_module_path, start=module_path.parent)
            )

            if not (module_parent_path / relative_path).resolve().exists():
                raise RuntimeError(
                    f"cannot find relative module path for {relative_path.resolve()}"
                )

            self.module_relative_path = relative_path

        self.module_name = self.__class__.__module__
        self.class_name = self.__class__.__qualname__
        self.raw_class = self.__class__

    def save_dict(self) -> dict:
        members: dict = {}

        for key in self.__dict__:
            data = self.__dict__[key]
            if isinstance(data, Serializable):
                members[key] = self.__dict__[key].save_dict()
            else:
                members[key] = copy.deepcopy(self.__dict__[key])

        return {
            "source": "hyperimpute",
            "data": members,
            "version": self.version(),
            "class_name": self.class_name,
            "class": self.raw_class,
            "module_name": self.module_name,
            "module_relative_path": self.module_relative_path,
        }

    def save(self) -> dict:
        return save(self.save_dict())

    @validate_arguments
    def save_to_file(self, path: Path) -> bytes:
        raise NotImplementedError()

    @staticmethod
    # @validate_arguments
    def load_dict(representation: dict) -> Any:
        if "source" not in representation or representation["source"] != "hyperimpute":
            raise ValueError("Invalid hyperimpute object")

        if representation["version"] != Serializable.version():
            raise RuntimeError(
                f"Invalid hyperimpute API version. Current version is {Serializable.version()}, but the object was serialized using version {representation['version']}"
            )

        if representation["module_relative_path"] is not None:
            module_path = module_parent_path / representation["module_relative_path"]

            if not module_path.exists():
                raise RuntimeError(f"Unknown module path {module_path}")

            spec = importlib.util.spec_from_file_location(
                representation["module_name"], module_path
            )

            if not isinstance(spec.loader, Loader):
                raise RuntimeError("invalid hyperimpute object type")

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        cls = representation["class"]

        obj = cls()

        obj_dict = {}
        for key in representation["data"]:
            val = representation["data"][key]

            if (
                isinstance(val, dict)
                and "source" in val
                and val["source"] == "hyperimpute"
            ):
                obj_dict[key] = Serializable.load_dict(val)
            else:
                obj_dict[key] = val

        obj.__dict__ = obj_dict

        return obj

    @staticmethod
    @validate_arguments
    def load(buff: bytes) -> Any:
        representation = load(buff)

        return Serializable.load_dict(representation)

    @staticmethod
    def version() -> str:
        "API version"
        return MAJOR_VERSION


def _add_version(obj: Any) -> Any:
    obj._serde_version = MAJOR_VERSION
    return obj


def _check_version(obj: Any) -> Any:
    local_version = obj._serde_version

    if not hasattr(obj, "_serde_version"):
        raise RuntimeError("Missing serialization version")

    if local_version != MAJOR_VERSION:
        raise ValueError(
            f"Serialized object mismatch. Current major version is {MAJOR_VERSION}, but the serialized object has version {local_version}."
        )


def save(model: Any) -> bytes:
    _add_version(model)
    return cloudpickle.dumps(model)


def load(buff: bytes) -> Any:
    obj = cloudpickle.loads(buff)
    _check_version(obj)
    return obj


def save_to_file(path: Union[str, Path], model: Any) -> Any:
    _add_version(model)
    with open(path, "wb") as f:
        return cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        obj = cloudpickle.load(f)
        _check_version(obj)
        return obj
