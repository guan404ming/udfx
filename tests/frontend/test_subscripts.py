from udfx.frontend import trace
from udfx.frontend.logic import conj, disj, neg
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.types import INT, STR

S = Col("s", STR)
OPEN = Lit(None, INT.or_none())


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


def test_nullable_container_faults_first() -> None:
    s = Col("s", STR.or_none())
    program = trace(first, {"s": s.type})
    missing = apply("is_none", s)
    (index,) = apply("__getitem__", s, Lit(0, INT)).op.faults(s, Lit(0, INT))
    assert program.fault == disj(missing, conj(neg(missing), index.when))
