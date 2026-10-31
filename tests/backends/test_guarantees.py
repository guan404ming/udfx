import json
import math
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.rules import Lowered, RuleSet
from udfx.frontend import trace
from udfx.ir import FALSE, Call, Col, Guarded, If, Node, lit
from udfx.ops import OpKey
from udfx.ops.registry import Op, register, scoped
from udfx.program import Program
from udfx.types import BOOL, FLOAT, INT, STR, Kind


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "variant",
    [
        "default",
        "different_default",
        "transformed",
        "guarded_input",
        "unproven",
        "inexact_false",
        "negative_zero",
    ],
)
def test_native_guarantee_preserves_branch_results(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    rules = RuleSet[Any]()
    monkeypatch.setattr(backend, "rules", rules)
    with scoped():
        validation = register(Op("custom_valid", (INT,), BOOL, lambda value: value >= 0))
        consumer = register(Op("custom_consumer", (INT,), INT, lambda value: value))
        shift = register(Op("custom_shift", (INT,), INT, lambda value: value + 1))

        @rules("custom_valid", Kind.INT, false_exact=variant != "inexact_false")
        def valid(value: Any) -> Lowered[Any]:
            return Lowered(value != backend.literal(lit(-1)), value >= backend.literal(lit(-1)))

        @rules(
            "custom_consumer",
            Kind.INT,
            proves=None if variant == "unproven" else lambda node: Call(validation, node.arguments),
        )
        def consumed(value: Any) -> Lowered[Any]:
            return Lowered(value, value >= backend.literal(lit(0)))

        rules("custom_shift", Kind.INT)(lambda value: value + backend.literal(lit(1)))
        schema = {"value": INT, "flag": BOOL, "input_exact": BOOL}
        source: Node = Col("value", INT)
        if variant == "guarded_input":
            source = Guarded(source, Col("input_exact", BOOL))
        output = FLOAT if variant == "negative_zero" else INT
        default_value = 0.0 if variant == "negative_zero" else 7
        inner_default_value = (
            -0.0 if variant == "negative_zero" else 8 if variant == "different_default" else 7
        )
        default, inner_default = lit(default_value), lit(inner_default_value)
        body: Node = If(Col("flag", BOOL), Call(consumer, (source,)), inner_default, output)
        if variant == "transformed":
            body = Call(shift, (body,))
        node = If(Call(validation, (source,)), body, default, output)
        lowered = backend.lower(node)
        rows = [
            (5, True, True),
            (5, False, True),
            (-2, True, True),
            (-2, False, True),
            (-1, True, True),
            (-1, True, False),
        ]
        exact = backend.exact_expression(lowered)
        flags = backend.evaluate([exact], schema, rows)[0]
        assert flags[3] is (variant == "default")
        if variant == "guarded_input":
            assert flags[-1] is False

        def reference(value: int, flag: bool, input_exact: bool) -> int | float:
            value = value + 2 if variant == "guarded_input" and not input_exact else value
            result = (value if flag else inner_default_value) if value >= 0 else default_value
            if variant == "transformed" and value >= 0:
                result += 1
            return float(result) if variant == "negative_zero" else result

        program = Program(reference, schema, node)
        expression = backend.compose(lowered, program)
        actual = backend.evaluate([expression], schema, rows)[0]
        expected = [reference(*row) for row in rows]
        assert actual == expected
        if variant == "negative_zero":
            assert isinstance(actual[3], float)
            assert math.copysign(1.0, actual[3]) == 1.0


@pytest.mark.parametrize(
    "backend",
    [
        value
        for value in backend_params()
        if value.id in {"polars", "duckdb", "spark", "spark-ansi-off"}
    ],
)
def test_array_decoder_proves_json_validation(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def packed(text: str) -> str:
        try:
            return json.dumps([value.lower() for value in json.loads(text)])
        except (ValueError, TypeError, AttributeError):
            return "[]"

    key = OpKey("json.valid", (Kind.STR,))
    original = backend.rules.get(key)
    assert original is not None

    def guarded(value: Any) -> Lowered[Any]:
        lowered = original(value)
        assert isinstance(lowered, Lowered)
        return replace(lowered, exact=backend.literal(FALSE))

    monkeypatch.setitem(backend.rules._rules, key, guarded)  # pyright: ignore[reportPrivateUsage]
    program = trace(packed, {"text": STR})
    lowered = backend.lower_program(program)
    rows = [
        ('["Alpha", "BETA"]',),
        ("[]",),
        ('["Alpha"] trailing',),
        ('["Alpha",]',),
        ('["\\u0041"]',),
        ('{"Alpha":1}',),
        ("[null]",),
        ('"AB"',),
        ("true trailing",),
        ("invalid",),
    ]
    flags = backend.evaluate([backend.exact_expression(lowered)], program.schema, rows)[0]
    assert flags[:4] == [True, True, False, False]
    expression = backend.compose(lowered, program)
    assert backend.evaluate([expression], program.schema, rows) == [program.run(rows)]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("variant", ["condition", "removed"])
def test_condition_guarantee_requires_retained_guard(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    rules = RuleSet[Any]()
    monkeypatch.setattr(backend, "rules", rules)
    with scoped():
        outer = register(Op("custom_outer", (INT,), BOOL, lambda value: value >= -1))
        valid = register(Op("custom_valid", (INT,), BOOL, lambda value: value >= 0))
        consumer = register(Op("custom_consumer", (INT,), INT, lambda value: value))
        positive = register(Op("custom_positive", (INT,), BOOL, lambda value: value > 0))
        rules("custom_outer", Kind.INT, false_exact=True)(
            lambda value: Lowered(backend.literal(lit(True)), value >= backend.literal(lit(-1)))
        )
        rules(
            "custom_valid",
            Kind.INT,
            false_exact=True,
            proves=(lambda node: Call(outer, node.arguments)) if variant == "removed" else None,
        )(
            lambda value: Lowered(
                value != backend.literal(lit(-1)),
                backend.literal(FALSE)
                if variant == "condition"
                else value >= backend.literal(lit(-1)),
            )
        )
        rules("custom_consumer", Kind.INT, proves=lambda node: Call(valid, node.arguments))(
            lambda value: Lowered(value, value >= backend.literal(lit(0)))
        )
        rules("custom_positive", Kind.INT)(lambda value: value > backend.literal(lit(0)))
        source = Col("value", INT)
        consumed = Call(consumer, (source,))
        body = (
            If(Call(positive, (consumed,)), lit(10), lit(20), INT)
            if variant == "condition"
            else If(Col("flag", BOOL), consumed, lit(7), INT)
        )
        node = If(Call(valid, (source,)), body, lit(7), INT)
        if variant == "removed":
            node = If(Call(outer, (source,)), node, lit(9), INT)

        def reference(value: int, flag: bool) -> int:
            if variant == "removed" and value < -1:
                return 9
            if value < 0:
                return 7
            if variant == "condition":
                return 10 if value > 0 else 20
            return value if flag else 7

        program = Program(reference, {"value": INT, "flag": BOOL}, node)
        lowered = backend.lower(node)
        rows = [(5, True), (0, False), (-2, False), (-1, False)]
        flags = backend.evaluate([backend.exact_expression(lowered)], program.schema, rows)[0]
        assert flags[0] is True
        assert flags[2] is False
        expression = backend.compose(lowered, program)
        assert backend.evaluate([expression], program.schema, rows) == [program.run(rows)]
