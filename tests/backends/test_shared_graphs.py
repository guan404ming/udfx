import math
from collections.abc import Callable
from typing import Any, cast

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.graph import lower_graph
from udfx.backends.hoisting import free, reads_input, size
from udfx.backends.indexing import matching
from udfx.backends.lowering_context import lowering_context
from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.frontend.core.tracing import joining_helpers
from udfx.frontend.flow import machine
from udfx.graphs import canonicalize
from udfx.ir import FALSE, TRUE, Call, Col, Comprehension, Element, If, Node, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import FLOAT, INT, STR, list_of

GRAPH_BACKENDS = [
    value for value in backend_params() if cast("Backend[Any]", value.values[0]).supports_graphs()
]


def peel(text: str) -> str:
    if text.endswith("ab"):
        return text[:-2]
    if text.endswith("x"):
        return text[:-1]
    return text


def pipeline(text: str) -> str:
    for _ in range(12):
        text = peel(text)
    return text


def decrease(value: int) -> int:
    return value - 1 if value > 0 else 0


def decreasing_length(text: str) -> int:
    value = len(text)
    for _ in range(16):
        value = decrease(value)
    return value


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_graph_bindings_preserve_integer_bounds(backend: Backend[Any]) -> None:
    with joining_helpers():
        program = trace(decreasing_length, {"text": STR})
    assert size(program.value) > 10_000
    lowered = backend.lower_program(program)
    assert lowered.exact is None
    rows = [("",), ("x" * 16,), ("x" * 19,)]
    assert backend.evaluate([backend.build(program)], program.schema, rows) == [[0, 0, 3]]


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_helper_pipeline_keeps_shared_graph(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(machine, "MAX_STEPS", 120)
    rows = [("ab" * 15,), ("abx" * 8,), ("",), ("é中abx",)]
    program = trace(pipeline, {"text": STR})
    assert size(program.value) > 10_000
    assert check_function(pipeline, program.schema, backend, rows) == []
    lowered = backend.lower_program(program)
    _, flags = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], program.schema, rows
    )
    assert flags == [True] * len(rows)


@pytest.mark.parametrize("analyze", [free, reads_input])
def test_scope_analysis_keeps_shared_nodes_bounded(
    analyze: Callable[[Node], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    left = right = Element("value", STR)
    for _ in range(40):
        left, right = apply("__add__", left, left), apply("__add__", right, right)
    compare = Call.__eq__
    visits = 0

    def checked(self: Call, other: object) -> bool:
        nonlocal visits
        visits += 1
        assert visits <= 200
        return compare(self, other)

    monkeypatch.setattr(Call, "__eq__", checked)
    assert analyze(left) == analyze(right)


def test_comprehension_matching_keeps_shared_nodes_bounded() -> None:
    element = Element("value", INT)
    value: Node = element
    for _ in range(40):
        value = apply("__add__", value, value)
    node = Comprehension(Col("values", list_of(INT)), element, value, TRUE, FALSE)
    visits = 0

    def accept(part: Node) -> bool:
        nonlocal visits
        visits += 1
        assert visits <= 43
        return part is element

    assert matching(node, accept) == frozenset({element})


def test_graph_keeps_intermediate_overflow() -> None:
    def cancelled(value: int) -> int:
        total = value
        for _ in range(16):
            total += total
        return total - total

    backend = PolarsBackend()
    rows = [(0,), (1,), (2**62,), (-(2**62),)]
    program = trace(cancelled, {"value": INT})
    assert check_function(cancelled, program.schema, backend, rows) == []
    lowered = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
        [True, True, False, False]
    ]


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_graph_preserves_uncaught_faults(backend: Backend[Any]) -> None:
    def squares(text: str) -> int:
        value = int(text)
        for _ in range(16):
            value *= value
        return value

    schema = {"text": STR.or_none()}
    rows = [("0",), ("1",), ("-1",), ("\u0661",), ("bad",), (None,)]
    assert size(trace(squares, schema).value) > 10_000
    assert check_function(squares, schema, backend, rows) == []


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_joined_helpers_preserve_first_exception(backend: Backend[Any]) -> None:
    def parsed(value: str) -> int:
        return int(value.split("=")[1])

    def mapped(text: str) -> list[int]:
        try:
            return [parsed(word) for word in text.split(";") if word != "skip"]
        except IndexError:
            return [-1]
        except ValueError:
            return [-2]

    schema = {"text": STR}
    rows = [("skip;a=1",), ("bad;a=no",), ("a=no;bad",), ("a=٣",), ("",)]
    with joining_helpers():
        program = trace(mapped, schema)
    value, fault = canonicalize((program.value, program.fault))
    with lowering_context(binds=False, graph=True):
        lowered = lower_graph(backend, value, {})
    assert fault == FALSE
    expression = (
        lowered.expr
        if lowered.exact is None
        else backend.fallback(lowered.expr, backend.exact_expression(lowered), program)
    )
    assert backend.evaluate([expression], schema, rows) == [[mapped(*row) for row in rows]]


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_graph_preserves_signed_zero(backend: Backend[Any]) -> None:
    condition = apply("__gt__", Col("value", FLOAT), lit(0.0))
    magnitude = apply("abs", Col("value", FLOAT))
    value = If(
        condition,
        apply("__mul__", lit(-0.0), magnitude),
        apply("__mul__", lit(0.0), magnitude),
        FLOAT,
    )
    for _ in range(14):
        value = apply("__mul__", value, apply("__add__", value, lit(1.0)))
    program = Program(lambda value: -0.0 if value > 0 else 0.0, {"value": FLOAT}, value, FALSE)
    actual = backend.evaluate([backend.build(program)], program.schema, [(1.0,), (-1.0,)])[0]
    assert all(isinstance(value, float) for value in actual)
    assert [math.copysign(1, value) for value in actual if isinstance(value, float)] == [-1, 1]
