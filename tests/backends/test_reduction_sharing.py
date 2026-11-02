from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.rules import Lowered
from udfx.ir import FALSE, TRUE, Call, Col, Comprehension, Element, lit
from udfx.ops import Op, apply, register
from udfx.ops.registry import scoped
from udfx.program import Program
from udfx.types import INT, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_literal_sum_shares_captured_guards(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def guarded(value: Any) -> Lowered[Any]:
        nonlocal calls
        calls += 1
        exact = value >= backend.literal(lit(0))
        return Lowered(backend.select(exact, value, backend.literal(lit(0))), exact)

    with scoped():
        operation = register(Op("guarded_integer", (INT,), INT, lambda value: value))
        monkeypatch.setitem(backend.rules._rules, operation.key, guarded)  # pyright: ignore[reportPrivateUsage]
        invariant = Call(operation, (Col("value", INT),))
        element = Element("element", INT)
        items = Comprehension(
            lit((1, 2, 3), list_of(INT)),
            element,
            apply("__add__", invariant, element),
            TRUE,
            FALSE,
        )
        value = apply("sum", items)
        lowered = backend.lower(value)
        assert calls == 1
        program = Program(lambda value: value * 3 + 6, {"value": INT}, value)
        rows = [(-2,), (0,), (2,)]
        assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
            [False, True, True]
        ]
        assert backend.evaluate([backend.build(program)], program.schema, rows) == [[0, 6, 12]]
