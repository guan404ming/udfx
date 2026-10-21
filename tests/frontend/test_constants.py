# ruff: noqa: PLW0108, SIM401

from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.logic import conjunction, negate
from udfx.ir import FALSE, Col, If, Lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import FLOAT, INT, STR, list_of

pytestmark = pytest.mark.usefixtures("stand_ins")

S = Col("s", STR)
A = Col("a", INT)

CODES = ["x", "y"]
RATES = {"EUR": 1.0, "USD": 2.0}
NESTED = {"a": [1]}


def in_tuple(s: str) -> bool:
    return s in ("a", "b")


def in_list(s: str) -> bool:
    return s in ["b", "a"]


def in_set(s: str) -> bool:
    return s in {"b", "a"}


def in_tuple_with_none(s: str | None) -> bool:
    return s in ("a", None)


def in_set_with_none(s: str | None) -> bool:
    return s in {None, "b", "a"}


def in_ints(a: int) -> bool:
    return a in (2, 1)


def in_global(s: str) -> bool:
    return s in CODES


def in_mixed(s: str) -> bool:
    return s in (1, "a")


def in_text(s: str, t: str) -> bool:
    return s in t


def returns_tuple(s: str) -> tuple[str, str]:
    return ("a", "b")


def rate(s: str) -> float:
    rates = {"EUR": 1.0, "USD": 2.0}
    return rates[s]


def single(s: str) -> int:
    return {"a": 1}[s]


def global_rate(s: str) -> float:
    return RATES[s]


def known_rate(s: str) -> float:
    return RATES["USD"]


def unknown_rate(s: str) -> float:
    return RATES["GBP"]


def nested(s: str) -> list[int]:
    return NESTED[s]


def test_tuple_constant_becomes_list_literal() -> None:
    members = Lit(("a", "b"), list_of(STR))
    assert trace(in_tuple, {"s": STR}).value == apply("__contains__", members, S)


def test_none_item_is_not_in_a_list() -> None:
    s = Col("s", STR.or_none())
    program = trace(in_tuple, {"s": s.type})
    assert program.value == apply("__contains__", Lit(("a", "b"), list_of(STR)), s)
    assert program.fault == FALSE


def test_none_item_in_text_raises() -> None:
    s = Col("s", STR.or_none())
    assert trace(in_text, {"s": s.type, "t": STR}).fault == apply("is_none", s)


@pytest.mark.parametrize(
    ("function", "items"),
    [(in_tuple_with_none, ("a", None)), (in_set_with_none, ("a", "b", None))],
    ids=["tuple", "set"],
)
def test_none_item_makes_a_nullable_item_list(
    function: Callable[[str | None], bool], items: tuple[str | None, ...]
) -> None:
    s = Col("s", STR.or_none())
    program = trace(function, {"s": s.type})
    assert program.value == apply("__contains__", Lit(items, list_of(STR.or_none())), s)
    assert program.fault == FALSE
    rows = [("a",), ("b",), ("c",), (None,)]
    assert check_function(function, {"s": s.type}, PolarsBackend(), rows) == []


def test_list_constant_keeps_order() -> None:
    members = Lit(("b", "a"), list_of(STR))
    assert trace(in_list, {"s": STR}).value == apply("__contains__", members, S)


def test_set_constant_is_sorted() -> None:
    members = Lit(("a", "b"), list_of(STR))
    assert trace(in_set, {"s": STR}).value == apply("__contains__", members, S)


def test_int_constant_container() -> None:
    members = Lit((2, 1), list_of(INT))
    assert trace(in_ints, {"a": INT}).value == apply("__contains__", members, A)


def test_global_list_becomes_list_literal() -> None:
    members = Lit(("x", "y"), list_of(STR))
    assert trace(in_global, {"s": STR}).value == apply("__contains__", members, S)


def test_mixed_container_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="expected a value"):
        trace(in_mixed, {"s": STR})


def test_constant_tuple_return_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="tuple returns are unsupported"):
        trace(returns_tuple, {"s": STR})


def test_dict_lookup_becomes_if_chain() -> None:
    euro = apply("__eq__", S, Lit("EUR", STR))
    dollar = apply("__eq__", S, Lit("USD", STR))
    expected = If(euro, Lit(1.0, FLOAT), Lit(2.0, FLOAT), FLOAT)
    for function in (rate, global_rate):
        program = trace(function, {"s": STR})
        assert program.value == expected
        assert program.fault == conjunction(negate(euro), negate(dollar))


def test_build_map_lookup() -> None:
    program = trace(single, {"s": STR})
    test = apply("__eq__", S, Lit("a", STR))
    assert program.value == Lit(1, INT)
    assert program.fault == negate(test)


def test_none_key_misses() -> None:
    s = Col("s", STR.or_none())
    program = trace(rate, {"s": s.type})
    euro = apply("__eq__", s, Lit("EUR", STR))
    dollar = apply("__eq__", s, Lit("USD", STR))
    assert program.fault == conjunction(negate(euro), negate(dollar))


def test_constant_key_is_looked_up_statically() -> None:
    assert trace(known_rate, {"s": STR}).value == Lit(2.0, FLOAT)
    with pytest.raises(Unsupported, match="never returns"):
        trace(unknown_rate, {"s": STR})


def test_non_scalar_dict_value_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="not a scalar constant"):
        trace(nested, {"s": STR})


def test_static_lookup_fault_is_total() -> None:
    def guarded(s: str) -> float:
        if s:
            return RATES["GBP"]
        return 0.0

    program = trace(guarded, {"s": STR})
    assert program.fault == apply("bool", S)
    assert program.value == Lit(0.0, FLOAT)


@pytest.mark.parametrize(
    "function",
    [
        lambda value: {"a": "b", "c": "d"}.get(value),
        lambda value: {"a": "b", "c": "d"}.get(value, "missing"),
        lambda value: {"a": "b", "c": "d"}.get(value, value),
        lambda value: {}.get(value),
        lambda value: {}.get(value, value),
        lambda value: value in {"a": "b", "c": "d"},
        lambda value: value not in {"a": "b", "c": "d"},
        lambda value: {"a": "b"}.get("missing", value),
        lambda value: {"a": "b"}.get("a", value),
    ],
)
def test_dict_methods(function: Callable[..., object]) -> None:
    schema = {"value": STR.or_none()}
    rows = [("a",), ("c",), ("missing",), (None,)]
    assert trace(function, schema).fault == FALSE
    assert check_function(function, schema, PolarsBackend(), rows) == []


def test_dict_membership_guards_lookup() -> None:
    def replace(value: str) -> str:
        mapping = {"a": "b", "c": "d"}
        return mapping[value] if value in mapping else value

    rows = [("a",), ("c",), ("missing",), (None,)]
    assert check_function(replace, {"value": STR.or_none()}, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda value: {"a": 1}.get(),  # pyright: ignore[reportCallIssue]
        lambda value: {"a": 1}.get(value, 1, 2),  # pyright: ignore[reportCallIssue]
    ],
)
def test_dict_get_arity(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match=r"dict.get takes one or two arguments"):
        trace(function, {"value": STR})
