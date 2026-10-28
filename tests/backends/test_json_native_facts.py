import json
from dataclasses import replace
from importlib import import_module
from typing import Any
from unittest.mock import Mock, patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.lowering_context import assuming_exact
from udfx.ir import Col, Guarded, If, lit
from udfx.ops import OpKey, apply
from udfx.program import Program
from udfx.semantics.assumptions import assuming
from udfx.semantics.json_limits import within_limits
from udfx.testing.json_traps import JSON_TEXT_TRAPS, JSON_TRAPS
from udfx.types import INT, STR, Kind


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"polars", "datafusion"}]
)
@pytest.mark.parametrize(
    ("name", "kind"), [("json.array_native", list), ("json.string_native", str)]
)
def test_valid_json_reuses_native_shape(backend: Backend[Any], name: str, kind: type) -> None:
    value = Col("value", STR)
    valid = apply("json.valid", value)
    node = If(valid, Guarded(lit(1), apply(name, value)), lit(0), INT)

    def selected(value: str) -> int:
        try:
            parsed = json.loads(value)
        except ValueError:
            return 0
        return 1 if isinstance(parsed, kind) else 2

    program = Program(selected, {"value": STR}, node)
    key = OpKey(name, (Kind.STR,))
    rewrite = backend.rules.rewrite_call
    with (
        patch.object(
            backend.rules,
            "rewrite_call",
            side_effect=lambda node: None if node.op.key == key else rewrite(node),
        ),
        capture_lowering() as before,
    ):
        backend.build(program)
    lookup = backend.rules.get
    checked = Mock(wraps=lookup(key))
    with (
        patch.object(
            backend.rules,
            "get",
            side_effect=lambda candidate: checked if candidate == key else lookup(candidate),
        ),
        capture_lowering() as after,
    ):
        expr = backend.build(program)
    checked.assert_not_called()
    rows = [
        (value,)
        for value in (
            *JSON_TEXT_TRAPS,
            *JSON_TRAPS,
            "[]",
            ' ["a", "é"] ',
            '"text"',
            "[1]",
            "[[1]]",
            "null",
            "true",
            "1",
            '["a"] []',
            '["a",]',
            r'["\u0061"]',
            r'"\ud800"',
            r'["a\n"]',
            r'"a\n"',
        )
    ]
    assert backend.evaluate([expr], program.schema, rows) == [program.run(rows)]
    assert before[-1].exact is not None and after[-1].exact is not None
    flags = backend.evaluate([before[-1].exact, after[-1].exact], program.schema, rows)
    assert flags[0] == flags[1]


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"polars", "datafusion"}]
)
@pytest.mark.parametrize("native", [False, True])
def test_shape_requires_a_true_native_validation(backend: Backend[Any], native: bool) -> None:
    value = Col("value", STR)
    valid = apply("json.valid", value)
    key = OpKey("json.array_native", (Kind.STR,))
    lookup = backend.rules.get
    checked = Mock(wraps=lookup(key))
    with (
        patch.object(
            backend.rules,
            "get",
            side_effect=lambda candidate: checked if candidate == key else lookup(candidate),
        ),
        assuming(valid, not native),
    ):
        if native:
            with assuming_exact(valid):
                backend.lower(apply(key.name, value))
        else:
            backend.lower(apply(key.name, value))
    checked.assert_called_once()


def rejected_scan(text: str) -> int:
    raise ValueError("JSON limit scan")


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"duckdb", "spark"}]
)
@pytest.mark.parametrize("unicode", [False, True])
def test_json_limits_skip_unneeded_scans(backend: Backend[Any], unicode: bool) -> None:
    primitives = import_module(f"udfx.backends.{backend.name}.json").NATIVE
    schema = {"text": STR}
    failure = backend.native_raise(Program(rejected_scan, schema, lit(0)))
    assert failure is not None
    primitives = (
        replace(primitives, longest_digits=lambda value: failure)
        if unicode
        else replace(primitives, text=replace(primitives.text, char_length=lambda value: failure))
    )
    text = backend.column(Col("text", STR))
    expr = within_limits(primitives, text, backend.literal(lit('""')))
    document = json.dumps("漢" * 2000 if unicode else "short", ensure_ascii=False)
    assert backend.evaluate([expr], schema, [(document,)]) == [[True]]
