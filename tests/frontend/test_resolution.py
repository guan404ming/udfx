import functools
import math
from types import FunctionType

import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.calls import invoke
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import FLOAT, STR

POLARS = PolarsBackend()

S = Col("s", STR.or_none())


class Tag(str):
    pass


TAG = Tag("x")

RATES = {"a": 1.0}


def lower(s: str) -> str:
    return s.lower()


@functools.wraps(lower)
def wrapped(*arguments: str) -> str:
    return lower(*arguments)


@functools.wraps(bool)
def fake_bool(value: object) -> bool:
    return True


def truthy(s: str | None) -> bool:
    return bool(s)


def calls_fake_bool(s: str | None) -> bool:
    return fake_bool(s)


def tagged(s: str) -> str:
    return TAG


def keyword_only(s: str, *, t: str) -> str:
    return s


def defaulted(s: str = "x") -> str:
    return s


def measure(s: str) -> int:
    return len(s)


def log(x: float) -> float:
    return math.log10(x)


def above_pi(x: float) -> bool:
    return x > math.pi


def real_part(x: float) -> float:
    return x.real


def get_rate(s: str) -> float | None:
    return RATES.get(s)


def test_module_function_resolves_by_identity() -> None:
    x = Col("x", FLOAT)
    assert trace(log, {"x": FLOAT}).value == apply("math.log10", x)
    assert trace(above_pi, {"x": FLOAT}).value == apply("__gt__", x, Lit(math.pi, FLOAT))


def test_other_attribute_loads_are_unsupported() -> None:
    with pytest.raises(Unsupported, match="attribute load real"):
        trace(real_part, {"x": FLOAT})
    with pytest.raises(Unsupported, match="attribute load get"):
        trace(get_rate, {"s": STR})


def test_parameters_ignore_wraps() -> None:
    with pytest.raises(Unsupported, match="variable or keyword-only"):
        trace(wrapped, {"s": STR})


def test_keyword_only_and_defaults_are_unsupported() -> None:
    with pytest.raises(Unsupported, match="keyword-only"):
        trace(keyword_only, {"s": STR, "t": STR})
    with pytest.raises(Unsupported, match="defaults"):
        trace(defaulted, {"s": STR})


def test_callables_resolve_by_identity() -> None:
    assert trace(truthy, {"s": STR.or_none()}).value == apply("bool", S)
    with pytest.raises(Unsupported, match="unknown callable"):
        trace(calls_fake_bool, {"s": STR.or_none()})


def test_builtins_come_from_the_function() -> None:
    empty = FunctionType(measure.__code__, {"__builtins__": {}})
    with pytest.raises(Unsupported, match="unknown global len"):
        trace(empty, {"s": STR})
    swapped = FunctionType(measure.__code__, {"__builtins__": {"len": bool}})
    assert trace(swapped, {"s": STR}).value == apply("bool", Col("s", STR))


def test_scalar_subclass_is_not_a_literal() -> None:
    with pytest.raises(Unsupported, match="expected a value"):
        trace(tagged, {"s": STR})


def test_none_argument_of_method_raises_type_error() -> None:
    t = Col("t", STR.or_none())
    _, faults = invoke(".find", [S, t], AttributeError)
    assert [fault.kind for fault in faults] == [AttributeError, TypeError]


@given(batches({"x": FLOAT}))
def test_module_function_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(log, {"x": FLOAT}, POLARS, rows) == []
    assert check_function(above_pi, {"x": FLOAT}, POLARS, rows) == []


@given(batches({"s": STR.or_none()}))
def test_builtin_bool_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(truthy, {"s": STR.or_none()}, POLARS, rows) == []
