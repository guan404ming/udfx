import functools
import math
from collections.abc import Callable
from types import FunctionType

import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.calls import invoke
from udfx.ir import FALSE, Col, Lit
from udfx.ops import Op, apply, register
from udfx.ops.registry import scoped
from udfx.testing import batches, check_function
from udfx.types import FLOAT, INT, STR, Type, list_of

POLARS = PolarsBackend()

S = Col("s", STR.or_none())

STRICT_ITEMS = list_of(STR)

OPTIONAL_ITEMS = list_of(STR.or_none())


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


def text_or_empty(x: int | None) -> str:
    try:
        return str(x)
    except TypeError:
        return ""


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


def split_on_none(s: str) -> list[str]:
    return s.split(None)


def test_none_argument_takes_the_nullable_parameter() -> None:
    program = trace(split_on_none, {"s": STR})
    assert program.value == apply(".split", Col("s", STR), Lit(None, STR.or_none()))
    assert program.fault == FALSE


def none_text() -> str:
    return str(None)


def equals_none(x: float | None) -> bool:
    return x == None  # noqa: E711


def differs_from_none(x: float | None) -> bool:
    return x != None  # noqa: E711


def test_literal_call_runs_python() -> None:
    assert trace(none_text, {}).value == Lit("None", STR)


@pytest.mark.parametrize("type", [INT.or_none(), FLOAT.or_none(), STR.or_none()], ids=str)
@pytest.mark.parametrize(
    ("function", "name"), [(equals_none, "__eq__"), (differs_from_none, "__ne__")]
)
def test_none_operand_takes_the_other_operand_kind(
    function: Callable[[float | None], bool], name: str, type: Type
) -> None:
    x = Col("x", type)
    program = trace(function, {"x": type})
    assert program.value == apply(name, x, Lit(None, type))
    assert program.fault == FALSE


@given(batches({"x": INT.or_none(), "s": STR}))
def test_none_operands_match_python(rows: list[tuple[object, ...]]) -> None:
    schema = {"x": INT.or_none()}
    single = [row[:1] for row in rows]
    assert check_function(equals_none, schema, POLARS, single) == []
    assert check_function(differs_from_none, schema, POLARS, single) == []
    assert check_function(split_on_none, {"s": STR}, POLARS, [row[1:] for row in rows]) == []


def is_zero(x: float) -> bool:
    return x == 0


def test_float_compares_equal_to_int() -> None:
    x = Col("x", FLOAT)
    assert trace(is_zero, {"x": FLOAT}).value == apply("__eq__", x, Lit(0, INT))


def test_str_of_none_does_not_raise() -> None:
    program = trace(text_or_empty, {"x": INT.or_none()})
    assert program.value == apply("str", Col("x", INT.or_none()))
    assert program.fault == FALSE


@given(batches({"x": FLOAT}))
def test_module_function_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(log, {"x": FLOAT}, POLARS, rows) == []
    assert check_function(above_pi, {"x": FLOAT}, POLARS, rows) == []


@given(batches({"s": STR.or_none()}))
def test_builtin_bool_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(truthy, {"s": STR.or_none()}, POLARS, rows) == []


def first_item(items: list[str | None]) -> str | None:
    return items[0]


def joined(items: list[str]) -> str:
    return ",".join(items)


def test_nullable_items_do_not_bind_to_strict_items() -> None:
    with scoped():
        register(Op(".probe", (STRICT_ITEMS,), INT, len))
        _, faults = invoke(".probe", [Col("items", STRICT_ITEMS)], AttributeError)
        assert faults == []
        with pytest.raises(Unsupported, match=r"no op \.probe\(list\[str\?\]\)"):
            invoke(".probe", [Col("items", OPTIONAL_ITEMS)], AttributeError)


@pytest.mark.parametrize(
    ("items", "result"), [(STRICT_ITEMS, STR), (OPTIONAL_ITEMS, STR.or_none())], ids=str
)
def test_item_keeps_its_nullability(items: Type, result: Type) -> None:
    assert trace(first_item, {"items": items}).result == result


def test_join_of_nullable_items_faults_on_none() -> None:
    items = Col("items", OPTIONAL_ITEMS)
    program = trace(joined, {"items": OPTIONAL_ITEMS})
    assert program.fault == apply("__contains__", items, Lit(None, STR.or_none()))
    assert trace(joined, {"items": STRICT_ITEMS}).fault == FALSE


@given(batches({"items": OPTIONAL_ITEMS}))
def test_nullable_items_match_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(first_item, {"items": OPTIONAL_ITEMS}, POLARS, rows) == []
    assert check_function(joined, {"items": OPTIONAL_ITEMS}, POLARS, rows) == []
