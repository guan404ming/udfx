import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.logic import conj, disj, neg
from udfx.ir import Col, If, Lit
from udfx.ops import apply
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


def in_ints(a: int) -> bool:
    return a in (2, 1)


def in_global(s: str) -> bool:
    return s in CODES


def in_mixed(s: str) -> bool:
    return s in (1, "a")


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


def test_constant_container_outside_in_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="constant container"):
        trace(returns_tuple, {"s": STR})


def test_dict_lookup_becomes_if_chain() -> None:
    euro = apply("__eq__", S, Lit("EUR", STR))
    dollar = apply("__eq__", S, Lit("USD", STR))
    expected = If(euro, Lit(1.0, FLOAT), Lit(2.0, FLOAT), FLOAT)
    for function in (rate, global_rate):
        program = trace(function, {"s": STR})
        assert program.value == expected
        assert program.fault == conj(neg(euro), neg(dollar))


def test_build_map_lookup() -> None:
    program = trace(single, {"s": STR})
    test = apply("__eq__", S, Lit("a", STR))
    assert program.value == Lit(1, INT)
    assert program.fault == neg(test)


def test_none_key_raises_key_error() -> None:
    s = Col("s", STR.or_none())
    program = trace(rate, {"s": s.type})
    euro = apply("__eq__", s, Lit("EUR", STR))
    dollar = apply("__eq__", s, Lit("USD", STR))
    missing = apply("is_none", s)
    miss = conj(neg(missing), conj(neg(euro), neg(dollar)))
    assert program.fault == disj(missing, miss)


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
