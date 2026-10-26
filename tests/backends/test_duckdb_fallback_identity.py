from dataclasses import replace

import duckdb
import pytest

from udfx.backends.duckdb.backend import fallback_name
from udfx.ir import Call, Col, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.types import STR

pytestmark = pytest.mark.xdist_group("duckdb")


def identity(text: str) -> str:
    return text


def test_fallback_name_keeps_shared_graph_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Call.__repr__
    visited = 0

    def bounded(node: Call) -> str:
        nonlocal visited
        visited += 1
        if visited > 200:
            pytest.fail("Fallback naming expands shared IR nodes")
        return original(node)

    monkeypatch.setattr(Call, "__repr__", bounded)
    value = Col("text", STR)
    for _ in range(40):
        value = apply("__add__", value, value)
    program = Program(identity, {"text": STR}, value)
    with duckdb.connect() as connection:
        first = fallback_name(program, connection)
        assert first == fallback_name(program, connection)


def test_fallback_identity_uses_values_instead_of_graph_sharing() -> None:
    text = Col("text", STR)
    shared = apply("__add__", text, lit("!"))
    duplicated = apply("__add__", Col("text", STR), lit("!"))
    program = Program(identity, {"text": STR}, apply("__add__", shared, shared))
    same = replace(program, value=apply("__add__", shared, duplicated))
    other = replace(program, value=apply("__add__", shared, lit("?")))
    failing = replace(program, fault=apply("__eq__", text, lit("bad")))
    with duckdb.connect() as connection:
        first = fallback_name(program, connection)
        assert first == fallback_name(same, connection)
        assert first != fallback_name(other, connection)
        assert first != fallback_name(failing, connection)
