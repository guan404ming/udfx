import inspect
from collections.abc import Callable

import pyarrow


def with_arity[Function: Callable[..., object]](function: Function, count: int) -> Function:
    parameters = [
        inspect.Parameter(f"value{index}", inspect.Parameter.POSITIONAL_ONLY)
        for index in range(count)
    ]
    function.__signature__ = inspect.Signature(parameters)
    return function


def batched(
    function: Callable[..., object], result: pyarrow.DataType, count: int
) -> Callable[..., pyarrow.Array]:
    def apply(*arrays: pyarrow.Array) -> pyarrow.Array:
        columns = [array.to_pylist() for array in arrays]
        return pyarrow.array([function(*row) for row in zip(*columns, strict=True)], result)

    return with_arity(apply, count)
