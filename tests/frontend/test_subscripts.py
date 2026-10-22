import dis
from types import ModuleType

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.logic import conjunction, disjunction, fold, negate
from udfx.frontend.opcodes import py312, py313, py314
from udfx.ir import FALSE, Col, Lit
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
    assert program.fault == disjunction(missing, conjunction(negate(missing), index.when))


def test_literal_slice_step() -> None:
    program = trace(lambda s: s[::-1], {"s": STR})
    assert program.value == apply("slice", S, OPEN, OPEN, Lit(-1, INT))


def test_dynamic_slice_step_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="slice step must be a literal"):
        trace(lambda s, step: s[::step], {"s": STR, "step": INT})


def test_zero_slice_step_fault() -> None:
    def caught(s: str) -> str:
        try:
            return s[::0]
        except ValueError:
            return "zero"

    assert trace(caught, {"s": STR}).value == Lit("zero", STR)


@pytest.mark.parametrize("normalizer", [py312, py313, py314], ids=["3.12", "3.13", "3.14"])
@pytest.mark.parametrize("step", [-2, -1, 2])
def test_slice_opcode_versions(
    monkeypatch: pytest.MonkeyPatch, normalizer: ModuleType, step: int
) -> None:
    def function(s: str) -> str:
        return s

    resume, load, returned = dis.get_instructions(function)
    if normalizer is py314:
        sliced = [
            load._replace(opname="LOAD_CONST", argval=slice(None, None, step)),
            load._replace(opname="BINARY_OP", argrepr="[]"),
        ]
    else:
        load = load._replace(opname="LOAD_FAST")
        sliced = [
            *(load._replace(opname="LOAD_CONST", argval=value) for value in (None, None, step)),
            load._replace(opname="BUILD_SLICE", arg=3, argval=3),
            load._replace(opname="BINARY_SUBSCR"),
        ]
    instructions = [resume, load, *sliced, returned]
    instructions = [item._replace(offset=2 * index) for index, item in enumerate(instructions)]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    monkeypatch.setattr("udfx.frontend.context.normalize", normalizer.normalize)
    assert trace(function, {"s": STR}).value == apply("slice", S, OPEN, OPEN, Lit(step, INT))


@pytest.mark.parametrize(
    ("pattern", "offset", "rewritten"),
    [("-", 1, True), ("ab", 2, False), ("aba", 3, False), ("", 0, False), ("ab", 1, False)],
)
def test_suffix_slice_reuses_split(pattern: str, offset: int, rewritten: bool) -> None:
    separator = Lit(pattern, STR)
    start = apply("__add__", apply(".rfind", S, separator), Lit(offset, INT))
    node = apply("slice", S, start, OPEN)
    expected = apply("last_split", S, separator)
    assert fold(node) == (expected if rewritten else node)


def test_suffix_slice_compiles_without_search() -> None:
    program = trace(lambda s: s[s.rfind("-") + 1 :], {"s": STR})
    assert program.value == apply("last_split", S, Lit("-", STR))


def test_explicit_split_has_first_item() -> None:
    program = trace(lambda s: s.split("::")[0], {"s": STR})
    assert program.fault == FALSE


def test_truthy_string_has_first_item() -> None:
    program = trace(lambda s: s[0] if s else "", {"s": STR})
    assert program.fault == FALSE
