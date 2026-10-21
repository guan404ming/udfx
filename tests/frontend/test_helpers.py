# ruff: noqa: PLW0108
import calendar
import functools
from collections.abc import Callable
from types import FunctionType

import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col, If, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import BOOL, INT, STR, list_of

POLARS = PolarsBackend()

S = Col("s", STR.or_none())
A = Col("a", INT)


def shout(s: str) -> str:
    def up(word: str) -> str:
        return word.upper()

    return up(s)


def above(a: int) -> bool:
    limit = 0

    def big(value: int) -> bool:
        return value > limit

    return big(a)


def late(a: int) -> bool:
    limit = 1

    def big(value: int) -> bool:
        return value > limit

    limit = 2
    return big(a)


def siblings(s: str) -> str:
    def up(word: str) -> str:
        return word.upper()

    def low(word: str) -> str:
        return up(word).lower()

    return low(s)


def guarded(s: str) -> str:
    def low(word: str) -> str:
        return word.lower()

    try:
        return low(s)
    except AttributeError:
        return "none"


def branching(s: str) -> str | None:
    def pick(word: str) -> str | None:
        if word:
            return word.upper()
        return None

    return pick(s)


def annotated(s: str | None) -> dict[str, list[str]] | str:
    def up(word: str | None) -> dict[str, list[str]] | str:
        return word.upper() if word else ""

    return up(s)


def recursive(a: int) -> int:
    def again(value: int) -> int:
        return again(value)

    return again(a)


def defaulted(s: str) -> str:
    def up(word: str = "x") -> str:
        return word.upper()

    return up(s)


def nonlocal_write(a: int) -> int:
    count = 0

    def bump(value: int) -> int:
        nonlocal count
        count = value
        return value

    return bump(a)


def keyword(s: str) -> str:
    def up(word: str) -> str:
        return word.upper()

    return up(word=s)


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


def dynamic_default(text: str) -> str:
    def helper(value: str = text) -> str:
        return value.lower()

    return helper()


def object_default(text: str, suffix: object = object()) -> str:
    return text


class Cleaner:
    def clean(self, text: str) -> str:
        return text.strip()


def make_above(limit: int) -> Callable[[int], bool]:
    def udf(a: int) -> bool:
        return a > limit

    return udf


def make_member(codes: list[str]) -> Callable[[str], bool]:
    def udf(s: str) -> bool:
        return s in codes

    return udf


def test_helper_is_inlined() -> None:
    assert trace(shout, {"s": STR}).value == apply(".upper", Col("s", STR))


@pytest.mark.parametrize("function", [global_helper, helper_chain])
def test_global_helpers_are_inlined(function: Callable[[str], str]) -> None:
    assert check_function(function, {"text": STR}, POLARS, [(" A ",), ("İ",)]) == []
    assert trace(function, {"text": STR}).value == apply(
        "__add__", apply(".lower", apply(".strip", Col("text", STR))), Lit("!", STR)
    )


def test_closure_helper_uses_its_own_closure() -> None:
    suffix = "!"

    def helper(text: str) -> str:
        return text + suffix

    assert trace(lambda text: helper(text), {"text": STR}).value == apply(
        "__add__", Col("text", STR), Lit("!", STR)
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
        sum,
        object,
    ],
)
def test_non_python_helpers_remain_unsupported(helper: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match="unknown callable"):
        trace(lambda text: helper(text), {"text": STR})


@pytest.mark.parametrize("helper", [dynamic_default, object_default])
def test_non_constant_helper_defaults_are_unsupported(helper: Callable[[str], str]) -> None:
    with pytest.raises(Unsupported, match="constant"):
        trace(lambda text: helper(text), {"text": STR})


def test_closure_reads_cell() -> None:
    assert trace(above, {"a": INT}).value == apply("__gt__", A, Lit(0, INT))


def test_closure_binds_late() -> None:
    assert trace(late, {"a": INT}).value == apply("__gt__", A, Lit(2, INT))


def test_sibling_helper_is_inlined() -> None:
    s = Col("s", STR)
    assert trace(siblings, {"s": STR}).value == apply(".lower", apply(".upper", s))


def test_helper_fault_reaches_caller_handler() -> None:
    program = trace(guarded, {"s": STR.or_none()})
    expected = If(apply("is_none", S), Lit("none", STR), apply(".lower", S), STR)
    assert program.value == expected
    assert program.fault == FALSE


def test_helper_outcomes_merge() -> None:
    program = trace(branching, {"s": STR.or_none()})
    nothing = Lit(None, STR.or_none())
    assert program.value == If(apply("bool", S), apply(".upper", S), nothing, STR.or_none())


def test_helper_annotations_are_evaluated_statically() -> None:
    program = trace(annotated, {"s": STR.or_none()})
    assert program.value == If(apply("bool", S), apply(".upper", S), Lit("", STR), STR)


def test_top_level_closure_constants() -> None:
    assert trace(make_above(5), {"a": INT}).value == apply("__gt__", A, Lit(5, INT))


def test_top_level_closure_container() -> None:
    members = Lit(("x", "y"), list_of(STR))
    program = trace(make_member(["x", "y"]), {"s": STR})
    assert program.value == apply("__contains__", members, Col("s", STR))
    assert program.result == BOOL


@pytest.mark.parametrize(
    ("function", "reason"),
    [
        (recursive, "recursive"),
        (nonlocal_write, "nonlocal"),
    ],
)
def test_unsupported_helpers(function: Callable[..., object], reason: str) -> None:
    schema = {"a": INT} if function in (recursive, nonlocal_write) else {"s": STR}
    with pytest.raises(Unsupported, match=reason):
        trace(function, schema)


@given(batches({"s": STR.or_none()}))
def test_string_helpers_match_python(rows: list[tuple[object, ...]]) -> None:
    for function in (shout, siblings, guarded, branching, annotated):
        assert check_function(function, {"s": STR.or_none()}, POLARS, rows) == []


@given(batches({"a": INT}))
def test_closures_match_python(rows: list[tuple[object, ...]]) -> None:
    for function in (above, late, make_above(5)):
        assert check_function(function, {"a": INT}, POLARS, rows) == []
