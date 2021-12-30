# stdlib
import time
from typing import Any, Callable, Type

# third party
import numpy as np
import pandas as pd

# hyperimpute absolute
import hyperimpute.logger as log


def expect_type_for(idx: int, dtype: Type) -> Callable:
    """Decorator used for argument type checking.

    Args:
        idx: which argument should be validated.
        dtype: expected data type.

    Returns:
        Callable: the decorator
    """

    def expect_type(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) <= idx:
                raise ValueError("expected parameter out of range.")
            if not isinstance(args[idx], dtype):
                err = f"unsupported data type {type(args[idx])} for args[{idx}]. Expecting {dtype}"
                log.critical(err)
                raise ValueError(err)

            return func(*args, **kwargs)

        return wrapper

    return expect_type


def expect_ndarray_for(idx: int) -> Callable:
    return expect_type_for(idx, np.ndarray)


def expect_dataframe_for(idx: int) -> Callable:
    return expect_type_for(idx, pd.DataFrame)


def benchmark(func: Callable) -> Callable:
    """Decorator used for function duration benchmarking. It is active only with DEBUG loglevel.

    Args:
        func: the function to be benchmarked.

    Returns:
        Callable: the decorator

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()

        log.debug(f"{func.__qualname__} took {round(end - start, 4)} seconds")
        return res

    return wrapper


__all__ = [
    "expect_type_for",
    "expect_ndarray_for",
    "expect_dataframe_for",
    "benchmark",
]
