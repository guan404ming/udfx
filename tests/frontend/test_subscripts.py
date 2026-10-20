import pytest

from udfx.frontend import trace
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.types import INT, STR

S = Col("s", STR)
OPEN = Lit(None, INT.or_none())

pytestmark = pytest.mark.usefixtures("stand_ins")


def first(s: str) -> str:
    return s[0]


def head(s: str, a: int) -> str:
    return s[:a]


def tail(s: str, a: int) -> str:
    return s[a:]


def middle(s: str, a: int) -> str:
    return s[1:a]


def test_subscript_uses_getitem() -> None:
    assert trace(first, {"s": STR}).value == apply("__getitem__", S, Lit(0, INT))


def test_slice_fills_open_bounds() -> None:
    a = Col("a", INT)
    schema = {"s": STR, "a": INT}
    assert trace(head, schema).value == apply("slice", S, OPEN, a)
    assert trace(tail, schema).value == apply("slice", S, a, OPEN)
    assert trace(middle, schema).value == apply("slice", S, Lit(1, INT), a)


def test_nullable_container_faults() -> None:
    program = trace(first, {"s": STR.or_none()})
    assert program.fault == apply("is_none", Col("s", STR.or_none()))
