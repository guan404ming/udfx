from udfx.frontend import trace
from udfx.ir import FALSE, Call, Col
from udfx.types import INT, STR


def test_method_call_becomes_op() -> None:
    program = trace(lambda s: s.lower(), {"s": STR})
    assert isinstance(program.value, Call)
    assert program.value.op.name == ".lower"
    assert program.value.args == (Col("s", STR),)
    assert program.fault == FALSE


def test_nullable_receiver_adds_fault() -> None:
    program = trace(lambda s: s.lower(), {"s": STR.or_none()})
    assert program.fault != FALSE


def test_binary_operator_becomes_op() -> None:
    program = trace(lambda a, b: a + b, {"a": INT, "b": INT})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "__add__"
