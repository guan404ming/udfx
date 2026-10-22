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
    "function",
    [
        lambda text: re.findall(r"\d+", text),
        lambda text: re.findall(r"(\d+)", text),
        lambda text: re.split(r"\d+", text),
    ],
    ids=["findall", "findall_group", "split"],
)
def test_regex_list_fallback(backend: Backend[Any], function: Callable[[str], list[str]]) -> None:
    schema = {"text": STR}
    rows = [("a12b34",), ("\x00a12b34",), ("",), ("\x00",)]
    program = trace(function, schema)
    lowered = backend.lower(program.value)
    assert lowered.exact is not None
    assert backend.evaluate([lowered.exact], schema, rows) == [[True, False, True, False]]
    assert backend.evaluate([backend.build(program)], schema, rows) == [
        [function(text) for (text,) in rows]
    ]


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
        assert str(lowered.expr).count("text") <= 3
