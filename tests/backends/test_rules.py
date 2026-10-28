from typing import Any

import polars as pl
import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.backends.registration import RuleSet
from udfx.backends.rules import fold_predicate
from udfx.ir import Call, Col, Guarded, Node, lit
from udfx.ops.registry import Op, apply, register, scoped
from udfx.semantics.facts import argument_length, result_type
from udfx.types import BOOL, INT, STR, Kind, Type, list_of


def test_rule_metadata_passes_literals_and_static_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = PolarsBackend()
    rules = RuleSet[pl.Expr]()
    monkeypatch.setattr(backend, "rules", rules)

    @rules(
        "custom_text",
        Kind.STR,
        Kind.INT,
        literals=(1,),
        facts={"maximum_length": argument_length, "result": result_type},
    )
    def native(
        text: pl.Expr, count: pl.Expr | int, *, maximum_length: int | None, result: Type
    ) -> pl.Expr:
        assert maximum_length == 3
        assert result == STR
        return text.str.slice(0, count)

    with scoped():
        op = register(Op("custom_text", (STR, INT), STR, lambda text, count: text[:count]))
        constant = backend.lower(Call(op, (lit("abc"), lit(2))))
        dynamic = backend.lower(Call(op, (lit("abc"), Col("width", INT))))
        assert backend.evaluate([constant.expr, dynamic.expr], {"width": INT}, [(1,), (2,)]) == [
            ["ab", "ab"],
            ["a", "ab"],
        ]


@pytest.mark.parametrize("backend", backend_params())
def test_literal_translation_needs_no_fallback(backend: Backend[Any]) -> None:
    node = apply(".translate", Col("text", STR), lit("ab"), lit(("b", "a"), list_of(STR)))
    assert backend.exact(backend.lower(node)) is None


def test_rewrite_runs_before_lowering_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = PolarsBackend()
    rules = RuleSet[pl.Expr]()
    monkeypatch.setattr(backend, "rules", rules)

    @rules.rewrite("custom_outer", Kind.STR)
    def collapse(node: Call) -> Node:
        inner = node.arguments[0]
        assert isinstance(inner, Call)
        return inner.arguments[0]

    with scoped():
        inner = register(Op("custom_inner", (STR,), STR, lambda value: value))
        outer = register(Op("custom_outer", (STR,), STR, lambda value: value))
        node = Call(outer, (Call(inner, (Col("text", STR),)),))
        lowered = backend.lower(node)
        assert lowered.exact is None
        assert backend.evaluate([lowered.expr], {"text": STR}, [("abc",)]) == [["abc"]]
    with pytest.raises(ValueError, match="duplicate rewrite"):
        rules.rewrite("custom_outer", Kind.STR)(collapse)


def test_rewrite_runs_before_predicate_folding() -> None:
    rules = RuleSet[pl.Expr]()

    @rules.rewrite("custom_pair", Kind.BOOL, Kind.BOOL)
    def expand(node: Call) -> Node:
        return apply("and", *node.arguments)

    with scoped():
        pair = register(Op("custom_pair", (BOOL, BOOL), BOOL, lambda left, right: left and right))
        first, second, third = (Col(name, BOOL) for name in ("first", "second", "third"))
        node = apply("and", first, Call(pair, (second, third)))
        expected = apply("and", apply("and", first, second), third)
        assert fold_predicate(node, rules.rewrite_call) == expected


@pytest.mark.parametrize("include_source", [False, True])
def test_rule_expressions_preserve_all_guards(
    monkeypatch: pytest.MonkeyPatch, include_source: bool
) -> None:
    backend = PolarsBackend()
    rules = RuleSet[pl.Expr]()
    monkeypatch.setattr(backend, "rules", rules)
    text = Col("text", STR)
    source = Guarded(text, Col("source", BOOL))

    @rules(
        "custom_source",
        Kind.STR,
        expressions={"source": lambda node: source if include_source else None},
    )
    def native(value: pl.Expr, *, source: pl.Expr | None) -> pl.Expr:
        return value if source is None else source

    with scoped():
        op = register(Op("custom_source", (STR,), STR, lambda value: value))
        node = Call(op, (Guarded(text, Col("original", BOOL)),))
        lowered = backend.lower(node, {text: pl.col("bound")})
        assert lowered.exact is not None
        schema = {"bound": STR, "original": BOOL, "source": BOOL}
        rows = [("a", True, True), ("b", False, True), ("c", True, False)]
        assert backend.evaluate([lowered.expr, lowered.exact], schema, rows) == [
            ["a", "b", "c"],
            [True, False, not include_source],
        ]
