import re
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "guarded_backends"),
    [
        (lambda text: re.findall(r"\d+", text), {"datafusion"}),
        (lambda text: re.findall(r"(\d+)", text), {"datafusion"}),
        (lambda text: re.findall(r"[a-z](\d+)", text), {"polars", "datafusion"}),
        (lambda text: re.split(r"\d+", text), {"polars", "duckdb", "datafusion", "spark"}),
    ],
    ids=["findall", "findall_group", "findall_partial_group", "split"],
)
def test_regex_list_guards(
    backend: Backend[Any], function: Callable[[str], list[str]], guarded_backends: set[str]
) -> None:
    schema = {"text": STR}
    rows = [("a12b34",), ("\x00a12b34",), ("",), ("\x00",)]
    program = trace(function, schema)
    lowered = backend.lower(program.value)
    guarded = backend.name in guarded_backends
    assert (lowered.exact is not None) == guarded
    _, flags = backend.evaluate(
        [backend.column(Col("text", STR)), backend.exact_expression(lowered)], schema, rows
    )
    assert flags == [True, not guarded, True, not guarded]
    assert backend.evaluate([backend.build(program)], schema, rows) == [
        [function(text) for (text,) in rows]
    ]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("pattern", ["(a$)", "(?m)(^a$)", r"(\ba\b)", "(?i)(a+)"])
def test_whole_capture_preserves_anchors_and_flags(backend: Backend[Any], pattern: str) -> None:
    def function(text: str) -> list[str]:
        return re.findall(pattern, text)

    rows = [("a",), ("a\n",), ("b\na\nb",), ("aA",), ("ba",)]
    program = trace(function, {"text": STR})
    assert backend.evaluate([backend.build(program)], program.schema, rows) == [program.run(rows)]


@pytest.mark.parametrize("backend", backend_params())
def test_regex_native_execution(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    for node in (
        apply("re.sub", text, lit(r"\d+"), lit(0), lit("#"), lit(0)),
        apply("re.findall", text, lit(r"\d+"), lit(0)),
        apply("re.split", text, lit(r"\d+"), lit(0), lit(0)),
        *(
            apply(f"re.{name}", text, lit(r"\d+"), lit(0))
            for name in ("search", "match", "fullmatch")
        ),
    ):
        lowered = backend.lower(node)
        _, flags = backend.evaluate(
            [backend.column(text), backend.exact_expression(lowered)],
            {"text": STR},
            [("12٣\U0001d7dc",), ("",)],
        )
        assert flags == [True, True]
        assert str(lowered.expr).count("text") <= (6 if backend.name == "spark" else 3)
        if node.op.name not in {"re.findall", "re.split"}:
            assert lowered.exact is None
