from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.rules import Lowered
from udfx.ir import Call, Col, lit
from udfx.ops import Op, OpKey, apply, register
from udfx.ops.registry import scoped
from udfx.program import Program
from udfx.types import FLOAT, Kind


@pytest.mark.parametrize("backend", backend_params())
def test_shared_arguments_keep_one_exactness_guard(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    absolute = backend.rules.get(OpKey("abs", (Kind.FLOAT,)))
    assert absolute is not None

    def guarded(value: Any) -> Lowered[Any]:
        return Lowered(absolute(value), value >= backend.literal(lit(0.0)))

    with scoped():
        operation = register(Op("guarded_value", (FLOAT,), FLOAT, lambda value: value))
        monkeypatch.setitem(backend.rules._rules, operation.key, guarded)  # pyright: ignore[reportPrivateUsage]
        source = Col("guard_source", FLOAT)
        value = Call(operation, (source,))
        total = apply("__add__", value, value)
        lowered = backend.lower(total)
        assert str(lowered.exact).count("guard_source") == 1
        program = Program(lambda guard_source: guard_source * 2, {"guard_source": FLOAT}, total)
        rows = [(2.0,), (0.0,), (-2.0,)]
        assert backend.evaluate([backend.exact_expression(lowered)], program.schema, rows) == [
            [True, True, False]
        ]
        assert backend.evaluate([backend.build(program)], program.schema, rows) == [
            [4.0, 0.0, -4.0]
        ]
