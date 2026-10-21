# ruff: noqa: E721, PLW0108

from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Lit
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type


@pytest.mark.parametrize(
    "function",
    [
        lambda value: type(value) is str,
        lambda value: type(value) is not str,
        lambda value: type(value) == str,
        lambda value: type(value) != str,
        lambda value: str == type(value),
        lambda value: isinstance(value, str),
        lambda value: isinstance(value, int),
        lambda value: isinstance(value, (int, float)),
        lambda value: isinstance(value, (str, (int, float))),
        lambda value: isinstance(value, ()),  # pyright: ignore[reportUnnecessaryIsInstance]
        lambda value: isinstance(value, type(None)),
    ],
)
@pytest.mark.parametrize(
    ("type", "values"),
    [(STR, ["", "a"]), (INT, [0, 1]), (FLOAT, [0.0, float("nan")]), (BOOL, [False, True])],
)
def test_type_check(function: Callable[..., object], type: Type, values: list[object]) -> None:
    program = trace(function, {"value": type})
    assert isinstance(program.value, Lit)
    assert program.fault == FALSE
    schema = {"value": type.or_none()}
    assert trace(function, schema).fault == FALSE
    assert (
        check_function(function, schema, PolarsBackend(), [(value,) for value in [*values, None]])
        == []
    )


@pytest.mark.parametrize(
    "function",
    [
        lambda value: type(value),
        lambda value: isinstance(value, "str"),
        lambda value: isinstance(value, (str, "int")),  # pyright: ignore[reportArgumentType]
        lambda value: type(value) < str,  # pyright: ignore[reportOperatorIssue]
        lambda value: type(value, (), {}),
        lambda value: isinstance(value),  # pyright: ignore[reportCallIssue]
    ],
)
def test_unsupported_type_check(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": STR})
