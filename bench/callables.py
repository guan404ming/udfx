import inspect
from collections.abc import Callable, Sequence

import pyarrow

from udfx.types import Type


def with_arity[Function: Callable[..., object]](function: Function, count: int) -> Function:
    parameters = [
        inspect.Parameter(f"value{index}", inspect.Parameter.POSITIONAL_ONLY)
        for index in range(count)
    ]
    function.__signature__ = inspect.Signature(parameters)
    return function


def null_boundary(
    function: Callable[..., object], parameters: Sequence[Type]
) -> Callable[..., object]:
    required = tuple(index for index, type in enumerate(parameters) if not type.nullable)
    if not required:
        return function

    def apply(*values: object) -> object:
        if any(values[index] is None for index in required):
            return None
        return function(*values)

    return with_arity(apply, len(parameters))


def batched(
    function: Callable[..., object], result: pyarrow.DataType, count: int
) -> Callable[..., pyarrow.Array]:
    def apply(*arrays: pyarrow.Array) -> pyarrow.Array:
        columns = [array.to_pylist() for array in arrays]
        return pyarrow.array([function(*row) for row in zip(*columns, strict=True)], result)

    return with_arity(apply, count)
