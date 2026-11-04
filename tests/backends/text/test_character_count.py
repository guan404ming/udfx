from copy import copy
from dataclasses import replace
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.frontend.core.comprehensions import character_count
from udfx.ir import FALSE, TRUE, Call, Col, Comprehension, Element, Lit, Node, lit
from udfx.ops import Op, apply, register
from udfx.ops.registry import scoped
from udfx.testing import check_function
from udfx.types import INT, STR, list_of


def counted(source: Node, members: tuple[str, ...] = ("a", "a", "😀", "\x00")) -> Call:
    element = Element("character", STR)
    values = Comprehension(
        apply("list", source),
        element,
        lit(1),
        apply("__contains__", Lit(members, list_of(STR)), element),
        FALSE,
    )
    return apply("sum", values)


def test_character_count_reuses_its_source_and_deduplicates_members() -> None:
    source = apply(".lower", Col("text", STR))
    result = character_count(counted(source))
    assert isinstance(result, Call)
    assert result.op.name == "__sub__"
    left, right = result.arguments
    assert isinstance(left, Call) and left.arguments[0] is source
    assert isinstance(right, Call)
    translated = right.arguments[0]
    assert isinstance(translated, Call)
    assert translated.op.name == ".translate"
    assert translated.arguments == (source, lit("a😀\x00"), Lit(("", "", ""), list_of(STR)))
    assert translated.arguments[0] is source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value", lit(2)),
        ("value", lit(True)),
        ("fault", TRUE),
        ("source", Col("items", list_of(STR))),
        ("condition", TRUE),
        (
            "condition",
            apply("__contains__", Lit(("aa",), list_of(STR)), Element("character", STR)),
        ),
        (
            "condition",
            apply("__contains__", Lit(("a",), list_of(STR)), Element("other", STR)),
        ),
        (
            "condition",
            apply("__contains__", Col("members", list_of(STR)), Element("character", STR)),
        ),
    ],
)
def test_character_count_rejects_unproved_comprehensions(field: str, value: Node) -> None:
    original = counted(Col("text", STR))
    values = original.arguments[0]
    assert isinstance(values, Comprehension)
    changed = replace(values, **{field: value})
    assert character_count(Call(original.op, (changed,))) is None


def test_character_count_rejects_nondeterministic_sources() -> None:
    with scoped():
        operation = register(Op("character_source", (), STR, lambda: "a", deterministic=False))
        assert character_count(counted(Call(operation, ()))) is None


def test_character_count_accepts_empty_members() -> None:
    result = character_count(counted(Col("text", STR), ()))
    assert isinstance(result, Call)
    assert result.type == INT


@pytest.mark.parametrize("backend", backend_params())
def test_character_count_keeps_values_faults_and_native_rows(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def count(text: str) -> int:
        return sum(1 for character in text.lower() if character in ["a", "a", "😀", "\x00"])

    schema = {"text": STR.or_none()}
    program = trace(count, schema)
    assert isinstance(program.value, Call) and program.value.op.name == "sum"
    assert character_count(program.value) is not None
    rewritten = backend.rules.rewrite_call(program.value)
    assert (rewritten is not None) == (backend.name != "duckdb")
    baseline = copy(backend)
    monkeypatch.setattr(baseline, "rules", copy(baseline.rules))
    rewrite = baseline.rules.rewrite_call
    monkeypatch.setattr(
        baseline.rules,
        "rewrite_call",
        lambda node: None if node.op.name == "sum" else rewrite(node),
    )
    before, after = baseline.lower_program(program), backend.lower_program(program)
    rows = [("",), ("aAa",), ("😀a\x00é\u0301",), ("other",), (None,)]
    flags = backend.evaluate(
        [baseline.predicate(before.exact), backend.predicate(after.exact)], schema, rows
    )
    assert [value is True for value in flags[0]] == [value is True for value in flags[1]]
    assert backend.evaluate([before.expr, after.expr], schema, rows[:-1]) == [[0, 3, 3, 0]] * 2
    assert check_function(count, schema, backend, rows) == []
