# ruff: noqa: PLW0108

from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Call, Col, lit
from udfx.ops import apply
from udfx.semantics.text import PREDICATES
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type


@pytest.mark.parametrize(
    ("function", "name"),
    [(lambda left, right: left & right, "and"), (lambda left, right: left | right, "or")],
)
def test_boolean_operators(function: Callable[..., object], name: str) -> None:
    schema = {"left": BOOL.or_none(), "right": BOOL.or_none()}
    program = trace(function, schema)
    assert program.value == apply(name, Col("left", schema["left"]), Col("right", schema["right"]))
    rows = [(left, right) for left in (False, True, None) for right in (False, True, None)]
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    "function", [lambda value: value & 1, lambda value: value | 1, lambda value: ~value]
)
def test_integer_bitwise_is_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": INT})


def local_membership(value: int) -> bool:
    choices = [1, 2, 3, 4]
    return value in choices


def test_extended_list_membership() -> None:
    program = trace(local_membership, {"value": INT})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "__contains__"
    assert (
        check_function(local_membership, {"value": INT}, PolarsBackend(), [(1,), (4,), (5,)]) == []
    )


@pytest.mark.parametrize("function", [lambda value: [1, value], lambda value: [1, 2, 3]])
def test_dynamic_or_returned_list(function: Callable[..., object]) -> None:
    assert check_function(function, {"value": INT}, PolarsBackend(), [(1,), (4,)]) == []


def test_dynamic_membership_tuple_is_unsupported() -> None:
    with pytest.raises(Unsupported):
        trace(lambda value: value in [1, value], {"value": INT})


@pytest.mark.parametrize(
    "function", [lambda value: min(value, 1, 2, -1), lambda value: max(value, 1, 2, -1)]
)
def test_many_extrema(function: Callable[..., object]) -> None:
    assert check_function(function, {"value": INT}, PolarsBackend(), [(0,), (5,), (-5,)]) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda first, second, third: min(first, second, third),
        lambda first, second, third: max(first, second, third),
    ],
)
@pytest.mark.parametrize("type", [FLOAT, STR])
def test_many_extrema_preserve_order(function: Callable[..., object], type: Type) -> None:
    schema: dict[str, Type] = dict.fromkeys(("first", "second", "third"), type)
    rows = (
        [(float("nan"), 1.0, 2.0), (1.0, float("nan"), 2.0), (-0.0, 0.0, 0.0)]
        if type == FLOAT
        else [("b", "a", "c")]
    )
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda value: min(value, 1, "a"),
        lambda value: max(value, 1, 2, key=str),
        lambda value: min(value),
    ],
)
def test_extrema_unsupported_forms(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": INT})


@pytest.mark.parametrize("name", ["startswith", "endswith"])
@pytest.mark.parametrize("prefixes", [("a", "b"), (), ("", "a")])
def test_tuple_affixes(name: str, prefixes: tuple[str, ...]) -> None:
    function = eval(f"lambda value: value.{name}({prefixes!r})")
    schema = {"value": STR.or_none()}
    assert (
        check_function(function, schema, PolarsBackend(), [("abc",), ("b",), ("",), (None,)]) == []
    )
    if not prefixes:
        assert trace(function, schema).value == lit(False)


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.startswith(("a", 1)),
        lambda value: value.endswith(["a", "b"]),
        lambda value: value.startswith((value, "a")),
        lambda value: value.endswith(("a", "b"), 1),
    ],
)
def test_affix_unsupported_forms(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": STR})


@pytest.mark.parametrize("method", PREDICATES)
def test_predicate_forms(method: str) -> None:
    function = eval(f"lambda value: value.{method}()")
    program = trace(function, {"value": STR.or_none()})
    assert isinstance(program.value, Call)
    assert program.value.op.name == f".{method}"
    assert program.fault == apply("is_none", Col("value", STR.or_none()))
    with pytest.raises(Unsupported):
        trace(eval(f"lambda value: value.{method}(1)"), {"value": STR})


@pytest.mark.parametrize("method", ["strip", "lstrip", "rstrip"])
@pytest.mark.parametrize("argument", ["None", "''", "'ab'", "characters"])
def test_strip_arguments(method: str, argument: str) -> None:
    function = eval(f"lambda value, characters: value.{method}({argument})")
    schema = {"value": STR.or_none(), "characters": STR.or_none()}
    rows = [("abca", "ab"), ("  a ", None), ("a", ""), (None, "a")]
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize("method", ["strip", "lstrip", "rstrip"])
def test_strip_unsupported_arguments(method: str) -> None:
    for arguments in ("1", "'a', 'b'"):
        with pytest.raises(Unsupported):
            trace(eval(f"lambda value: value.{method}({arguments})"), {"value": STR})


@pytest.mark.parametrize("method", ["title", "capitalize", "swapcase", "casefold"])
def test_case_mapping_forms(method: str) -> None:
    function = eval(f"lambda value: value.{method}()")
    program = trace(function, {"value": STR.or_none()})
    assert isinstance(program.value, Call)
    assert program.value.op.name == f".{method}"
    assert program.fault == apply("is_none", Col("value", STR.or_none()))
    with pytest.raises(Unsupported):
        trace(eval(f"lambda value: value.{method}('a')"), {"value": STR})


@pytest.mark.parametrize("method", ["zfill", "center", "ljust", "rjust"])
@pytest.mark.parametrize("width", ["5", "width"])
def test_padding_width_forms(method: str, width: str) -> None:
    function = eval(f"lambda value, width: value.{method}({width})")
    schema = {"value": STR.or_none(), "width": INT.or_none()}
    rows = [("-1", 4), ("a", -1), ("", 0), (None, 2), ("a", None)]
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize("method", ["center", "ljust", "rjust"])
@pytest.mark.parametrize("fill", ["*", "😀", "", "ab"])
def test_literal_fill_forms(method: str, fill: str) -> None:
    namespace: dict[str, object] = {}
    exec(
        f"def padded(value, width):\n"
        f"    try:\n        return value.{method}(width, {fill!r})\n"
        "    except TypeError:\n        return 'invalid'\n",
        namespace,
    )
    function = namespace["padded"]
    assert callable(function)
    schema = {"value": STR.or_none(), "width": INT.or_none()}
    rows = [("a", 4), ("abc", 1), (None, 2), ("a", None)]
    assert check_function(function, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize("method", ["center", "ljust", "rjust"])
def test_column_fill_is_unsupported(method: str) -> None:
    with pytest.raises(Unsupported, match="fill character must be a literal"):
        trace(eval(f"lambda value, fill: value.{method}(5, fill)"), {"value": STR, "fill": STR})


@pytest.mark.parametrize("method", ["zfill", "center", "ljust", "rjust"])
@pytest.mark.parametrize("arguments", ["", "1.5", "1, '*', '*'"])
def test_padding_unsupported_arguments(method: str, arguments: str) -> None:
    with pytest.raises(Unsupported):
        trace(eval(f"lambda value: value.{method}({arguments})"), {"value": STR})
