from typing import Any

import polars as pl
import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.backends.registration import RuleSet
from udfx.ir import Call, Col, lit
from udfx.ops.registry import Op, apply, register, scoped
from udfx.semantics.facts import argument_length, result_type
from udfx.types import INT, STR, Kind, Type, list_of


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
