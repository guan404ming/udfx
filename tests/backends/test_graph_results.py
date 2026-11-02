from typing import Any, cast

import pytest
from conftest import backend_params

from udfx.backends import program_lowering
from udfx.backends.base import Backend
from udfx.backends.graph_cost import graph_cost
from udfx.backends.rules import Lowered
from udfx.frontend import trace
from udfx.graphs import canonicalize
from udfx.ir import Call, Col, Node, lit
from udfx.ops import Op, OpKey, apply, register
from udfx.ops.registry import scoped
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import FLOAT, STR, Kind, list_of

GRAPH_BACKENDS = [
    value for value in backend_params() if cast("Backend[Any]", value.values[0]).supports_graphs()
]


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_graph_consumes_guarded_result_once(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(program_lowering, "EXPANDED_LIMIT", 100)
    absolute = backend.rules.get(OpKey("abs", (Kind.FLOAT,)))
    assert absolute is not None

    def guarded(value: Any) -> Lowered[Any]:
        return Lowered(absolute(value), value >= backend.literal(lit(0.0)))

    with scoped():
        operation = register(
            Op("guarded_identity", (FLOAT,), FLOAT, lambda value: value if value else 0.0)
        )
        monkeypatch.setitem(backend.rules._rules, operation.key, guarded)  # pyright: ignore[reportPrivateUsage]
        value: Node = Call(operation, (Col("value", FLOAT),))
        for _ in range(8):
            value = apply("__add__", value, value)
        program = Program(lambda value: (value if value else 0.0) * 256, {"value": FLOAT}, value)
        lowered = backend.lower_program(program)
        packed = lowered.consume(
            lambda part: backend.record([part.expr, backend.exact_expression(part)])
        )
        assert str(packed).lower().count("abs(") == 1
        rows = [(1.0,), (0.0,), (-1.0,)]
        assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
            [True, True, False]
        ]
        expression = backend.compose(lowered, program)
        if backend.name == "duckdb":
            assert str(expression).lower().count("abs(") == 1
        assert backend.evaluate([expression], program.schema, rows) == [[256.0, 0.0, -256.0]]


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_graph_shares_fault_roots_without_losing_native_rows(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def repeated(text: str) -> int:
        value = int(text)
        for _ in range(6):
            value += value
        return value

    schema = {"text": STR.or_none()}
    program = trace(repeated, schema)
    rows = [("5",), ("-7",), ("0",), ("\u0661",)]
    original = backend.lower_program(program)
    expected = backend.evaluate([backend.exact_expression(original)], schema, rows)
    monkeypatch.setattr(program_lowering, "EXPANDED_LIMIT", 100)
    shared = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(shared)], schema, rows) == expected
    assert check_function(repeated, schema, backend, [*rows, ("bad",), (None,)]) == []


@pytest.mark.parametrize("backend", GRAPH_BACKENDS)
def test_exact_graph_needs_no_result_binding(backend: Backend[Any]) -> None:
    value: Node = Col("value", FLOAT)
    for _ in range(14):
        value = apply("__add__", value, value)
    program = Program(lambda value: value * 16_384, {"value": FLOAT}, value)
    lowered = backend.lower_program(program)
    assert lowered.exact is None
    assert lowered.overflow is None
    assert lowered.prepare is None


@pytest.mark.parametrize(
    "backend",
    [value for value in GRAPH_BACKENDS if value.id in {"duckdb", "spark", "spark-ansi-off"}],
)
def test_graph_native_fault_consumes_result(backend: Backend[Any]) -> None:
    def reference(value: float) -> float:
        if value < 0:
            raise ValueError("negative input")
        return abs(value) * 16_384

    source = Col("value", FLOAT)
    value: Node = apply("abs", source)
    for _ in range(14):
        value = apply("__add__", value, value)
    program = Program(reference, {"value": FLOAT}, value, apply("__lt__", source, lit(0.0)))
    lowered = backend.lower_program(program)
    assert lowered.exact is None
    assert lowered.prepare is not None
    assert backend.evaluate([backend.exact_expression(lowered)], program.schema, [(-1.0,)]) == [
        [True]
    ]
    expression = backend.compose(lowered, program)
    assert str(expression).lower().count("abs(") == 1
    assert backend.evaluate([expression], program.schema, [(0.0,), (1.0,)]) == [[0.0, 16_384.0]]
    with pytest.raises(ValueError, match="negative input"):
        backend.evaluate([expression], program.schema, [(-1.0,)])


@pytest.mark.parametrize(
    "backend", [value for value in GRAPH_BACKENDS if value.id in {"spark", "spark-ansi-off"}]
)
def test_optional_graph_keeps_guarded_capturing_comprehension_native(backend: Backend[Any]) -> None:
    def captured(text: str, items: list[str]) -> list[str]:
        prefix = text
        for _ in range(12):
            prefix = prefix.strip()
        return [prefix, prefix] + [prefix + str(int(word)) for word in items]

    schema = {"text": STR, "items": list_of(STR)}
    program = trace(captured, schema)
    ordinary, shared = graph_cost(canonicalize((program.value, program.fault)))
    assert shared < ordinary
    lowered = backend.lower_program(program)
    rows = [(" A ", ["1", "2"]), ("ß", ["\u0662"]), ("A", [])]
    native, exact = backend.evaluate(
        [lowered.expr, backend.exact_expression(lowered)], schema, rows
    )
    assert exact == [True, True, True]
    assert native == program.run(rows)
