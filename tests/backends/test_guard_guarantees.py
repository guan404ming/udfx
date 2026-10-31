import json
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.guarantees import Guarantee
from udfx.backends.rules import Lowered, RuleSet
from udfx.frontend import trace
from udfx.ir import FALSE, Call, Col, Guarded, If, lit
from udfx.ops import OpKey
from udfx.ops.registry import Op, register, scoped
from udfx.program import Program
from udfx.types import BOOL, INT, STR, Kind


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("variant", ["false_default", "true_default", "inexact"])
def test_checked_predicate_guarantee(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    rules = RuleSet[Any]()
    monkeypatch.setattr(backend, "rules", rules)
    with scoped():
        valid = register(Op("custom_valid", (INT,), BOOL, lambda value: value >= 0))
        positive = register(
            Op(
                "custom_positive",
                (INT,),
                BOOL,
                lambda value: value < 0 if variant == "true_default" else value > 0,
            )
        )
        source = Col("value", INT)
        predicate = Call(valid, (source,))
        rules("custom_valid", Kind.INT, false_exact=True)(
            lambda value: Lowered(value != backend.literal(lit(-1)), backend.literal(FALSE))
        )
        rules("custom_positive", Kind.INT)(
            lambda value: Lowered(
                value < backend.literal(lit(0))
                if variant == "true_default"
                else value > backend.literal(lit(0)),
                backend.column(Col("exact", BOOL)) if variant == "inexact" else None,
                guarantees=(Guarantee(predicate, lit(variant == "true_default")),),
            )
        )
        guarded = Guarded(source, Call(positive, (source,)))
        node = If(predicate, guarded, lit(7), INT)

        def reference(value: int, exact: bool) -> int:
            return value if value >= 0 else 7

        program = Program(reference, {"value": INT, "exact": BOOL}, node)
        lowered = backend.lower(node)
        rows = [(5, True), (5, False), (-2, True), (-1, True)]
        flags = backend.evaluate([backend.exact_expression(lowered)], program.schema, rows)[0]
        assert flags[0] is (variant != "true_default")
        assert flags[1] is (variant == "false_default")
        assert flags[2] is False
        expression = backend.compose(lowered, program)
        assert backend.evaluate([expression], program.schema, rows) == [program.run(rows)]


@pytest.mark.parametrize(
    "backend",
    [value for value in backend_params() if value.id in {"duckdb", "spark", "spark-ansi-off"}],
)
def test_checked_json_shape_proves_validity(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def count(text: str) -> int:
        try:
            return len(json.loads(text))
        except (ValueError, TypeError):
            return -1

    key = OpKey("json.valid", (Kind.STR,))
    original = backend.rules.get(key)
    assert original is not None

    def guarded(value: Any) -> Lowered[Any]:
        lowered = original(value)
        assert isinstance(lowered, Lowered)
        return replace(lowered, exact=backend.literal(FALSE))

    monkeypatch.setitem(backend.rules._rules, key, guarded)  # pyright: ignore[reportPrivateUsage]
    program = trace(count, {"text": STR})
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
