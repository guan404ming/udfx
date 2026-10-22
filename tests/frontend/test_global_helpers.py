# ruff: noqa: PLW0108
import calendar
import functools
from collections.abc import Callable
from types import FunctionType

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import INT, STR

from .test_helpers import POLARS, defaulted, recursive


def clean_text(text: str, suffix: str = "!") -> str:
    return text.strip().lower() + suffix


def global_helper(text: str) -> str:
    return clean_text(text)


def helper_chain(text: str) -> str:
    return global_helper(text)


def global_recursive(value: int) -> int:
    return global_recursive(value)


def constant_recursion(value: int) -> int:
    if value:
        return constant_recursion(value - 1)
    return 7


class Cleaner:
    def clean(self, text: str) -> str:
        return text.strip()


@pytest.mark.parametrize("function", [global_helper, helper_chain])
def test_global_helpers_are_inlined(function: Callable[[str], str]) -> None:
    assert check_function(function, {"text": STR}, POLARS, [(" A ",), ("İ",)]) == []
    assert trace(function, {"text": STR}).value == apply(
        "__add__", apply(".strip", apply(".lower", Col("text", STR))), Lit("!", STR)
    )


def test_library_helper_is_inlined() -> None:
    def function(year: int) -> bool:
        return calendar.isleap(year)

    assert check_function(function, {"year": INT}, POLARS, [(1900,), (2000,), (2024,)]) == []


def test_helper_uses_its_own_builtins() -> None:
    def measure(text: str) -> int:
        return len(text)

    helper = FunctionType(measure.__code__, {"__builtins__": {"len": bool}})
    assert trace(lambda text: helper(text), {"text": STR}).value == apply("bool", Col("text", STR))


def test_helper_with_unsupported_global_names_the_value() -> None:
    helper = FunctionType(global_helper.__code__, {"clean_text": Cleaner()})
    with pytest.raises(Unsupported, match="unknown callable a Cleaner object"):
        trace(lambda text: helper(text), {"text": STR})


def test_helper_defaults_fill_missing_arguments() -> None:
    assert trace(defaulted, {"s": STR}).value == apply(".upper", Col("s", STR))
    assert trace(lambda: clean_text(" A "), {}).value == Lit("a!", STR)


def test_constant_recursion_finishes_before_depth_limit() -> None:
    assert trace(lambda: constant_recursion(3), {}).value == Lit(7, INT)


@pytest.mark.parametrize("function", [global_recursive, recursive])
def test_recursion_stops_at_depth_limit(function: Callable[[int], int]) -> None:
    with pytest.raises(Unsupported, match="recursive helper"):
        trace(lambda value: function(value), {"value": INT})


@pytest.mark.parametrize(
    "helper",
    [
        functools.partial(clean_text, suffix="!"),
        Cleaner().clean,
        "prefix".replace,
        str.title,
        iter,
        object,
    ],
)
def test_non_python_helpers_remain_unsupported(helper: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match="unknown callable"):
        trace(lambda text: helper(text), {"text": STR})
