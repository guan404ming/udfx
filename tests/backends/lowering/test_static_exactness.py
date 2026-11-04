import math
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.lowering import programs as program_lowering
from udfx.backends.rules import Lowered
from udfx.frontend import trace
from udfx.ir import Call, Col, If, Node, lit
from udfx.ops import Op, apply, register
from udfx.ops.registry import scoped
from udfx.program import Program
from udfx.testing import check_function
from udfx.testing.equivalence import check_op_batches
from udfx.types import BOOL, FLOAT, STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_graph_skips_preparation(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not backend.supports_graphs():
        pytest.skip("Graph lowering is unavailable")
    with scoped():
        operation = register(Op("inexact_identity", (FLOAT,), FLOAT, lambda value: value))
        monkeypatch.setitem(
            backend.rules._rules,  # pyright: ignore[reportPrivateUsage]
            operation.key,
            lambda value: Lowered(value, False),
        )
        value: Node = Call(operation, (Col("value", FLOAT),))
        for _ in range(14):
            value = apply("__add__", value, value)
        program = Program(lambda value: value * 16384, {"value": FLOAT}, value)
        lowered = backend.lower_program(program)
        assert lowered.exact is False

        def forbidden(finish: Any) -> Any:
            pytest.fail("A statically inexact graph was prepared")

        expression = backend.compose(replace(lowered, prepare=forbidden), program)
        assert backend.evaluate([expression], program.schema, [(1.0,), (-2.0,)]) == [
            [16384.0, -32768.0]
        ]


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_branch_preserves_native_rows(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    with scoped():
        operation = register(Op("inexact_identity", (FLOAT,), FLOAT, lambda value: value))
        monkeypatch.setitem(
            backend.rules._rules,  # pyright: ignore[reportPrivateUsage]
            operation.key,
            lambda value: Lowered(value, False),
        )
        value = Col("value", FLOAT)
        program = Program(
            lambda pending, value: value,
            {"pending": BOOL, "value": FLOAT},
            If(Col("pending", BOOL), Call(operation, (value,)), value, FLOAT),
        )
        lowered = backend.lower_program(program)
        rows = [(False, 1.0), (True, 2.0)]
        assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
            [True, False]
        ]
        assert backend.evaluate([backend.build(program)], program.schema, rows) == [[1.0, 2.0]]


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_keeps_caught_faults(backend: Backend[Any]) -> None:
    def caught(value: float) -> float:
        try:
            return math.log(value, 2.0)
        except ValueError:
            return -1.0

    rows = [(-1.0,), (0.0,), (1.0,), (4.0,), (None,)]
    assert check_function(caught, {"value": FLOAT.or_none()}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_keeps_native_fault_replay(backend: Backend[Any]) -> None:
    def selected(value: float) -> float:
        if value > 0:
            return math.log(value, 2.0)
        return math.sqrt(value)

    schema = {"value": FLOAT}
    program = trace(selected, schema)
    lowered = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(lowered)], schema, [(0.0,), (4.0,)]) == [
        [True, False]
    ]
    assert check_function(selected, schema, backend, [(-1.0,), (0.0,), (4.0,)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_keeps_nullable_branch_masks(backend: Backend[Any]) -> None:
    def selected(pending: bool, value: float) -> float:
        if pending:
            return math.log(value, 2.0)
        return value

    schema = {"pending": BOOL.or_none(), "value": FLOAT.or_none()}
    program = trace(selected, schema)
    lowered = backend.lower_program(program)
    rows = [(None, None), (False, 1.0), (True, 4.0), (None, 4.0), (True, None)]
    assert backend.evaluate([backend.exact_expression(lowered)], schema, rows) == [
        [True, True, False, True, False]
    ]
    assert check_function(selected, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("graph", [False, True])
def test_static_false_comprehension_keeps_empty_rows(
    backend: Backend[Any], graph: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if graph:
        if not backend.supports_graphs():
            pytest.skip("Graph lowering is unavailable")
        monkeypatch.setattr(program_lowering, "size", lambda node: program_lowering.EXPANDED_LIMIT)

    def logarithms(values: list[float]) -> list[float]:
        return [math.log(value, 2.0) for value in values]

    schema = {"values": list_of(FLOAT)}
    program = trace(logarithms, schema)
    lowered = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(lowered)], schema, [([],), ([4.0],)]) == [
        [True, False]
    ]
    assert check_function(logarithms, schema, backend, [([],), ([4.0],), ([-1.0],), ([0.0],)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_static_false_does_not_execute_overflowing_native_branch(backend: Backend[Any]) -> None:
    def reference(value: float) -> float:
        return math.log(value, 2.0)

    program = trace(reference, {"value": FLOAT})
    lowered = backend.lower_program(program)
    assert lowered.exact is False
    poisoned = replace(lowered, overflow=backend.literal(lit(True)))
    expression = backend.compose(poisoned, program)
    assert backend.evaluate([expression], program.schema, [(4.0,)]) == [[2.0]]


@pytest.mark.parametrize("backend", backend_params())
def test_conformance_normalizes_static_false(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    with scoped():
        operation = register(Op("inexact_identity", (FLOAT,), FLOAT, lambda value: value))
        monkeypatch.setitem(
            backend.rules._rules,  # pyright: ignore[reportPrivateUsage]
            operation.key,
            lambda value: Lowered(value, False),
        )
        assert check_op_batches(operation, backend, [([(1.0,), (-2.0,)], {})]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_graph_keeps_mixed_exactness_and_overflow_fields(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not backend.supports_graphs():
        pytest.skip("Graph lowering is unavailable")
    monkeypatch.setattr(program_lowering, "size", lambda node: program_lowering.EXPANDED_LIMIT)

    def reference(pending: bool, value: float) -> float:
        if not pending and value > 10:
            raise ValueError("large")
        return value * 2

    with scoped():
        pending = register(Op("pending_identity", (FLOAT,), FLOAT, lambda value: value))
        dynamic = register(Op("dynamic_identity", (FLOAT,), FLOAT, lambda value: value))
        monkeypatch.setitem(
            backend.rules._rules,  # pyright: ignore[reportPrivateUsage]
            pending.key,
            lambda value: Lowered(value, False, backend.literal(lit(True))),
        )
        monkeypatch.setitem(
            backend.rules._rules,  # pyright: ignore[reportPrivateUsage]
            dynamic.key,
            lambda value: Lowered(
                value, value >= backend.literal(lit(0.0)), value > backend.literal(lit(10.0))
            ),
        )
        value = Col("value", FLOAT)
        first, second = Call(pending, (value,)), Call(dynamic, (value,))
        node = If(
            Col("pending", BOOL),
            apply("__add__", first, first),
            apply("__add__", second, second),
            FLOAT,
        )
        program = Program(reference, {"pending": BOOL, "value": FLOAT}, node)
        lowered = backend.lower_program(program)
        rows = [(True, 11.0), (False, 2.0), (False, -2.0)]
        assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
            [False, True, False]
        ]
        expression = backend.build(program)
        assert backend.evaluate([expression], program.schema, rows) == [[22.0, 4.0, -4.0]]
        with pytest.raises(ValueError):
            backend.evaluate([expression], program.schema, [(False, 11.0)])


@pytest.mark.parametrize("backend", backend_params())
def test_wide_literal_padding_is_static_false(backend: Backend[Any]) -> None:
    def padded(text: str) -> str:
        return text.ljust(2**21)

    assert backend.lower_program(trace(padded, {"text": STR})).exact is False
