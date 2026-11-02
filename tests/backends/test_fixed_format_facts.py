from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.rules import Lowered
from udfx.frontend import trace
from udfx.ir import Call, Col, Element, Lit, Node, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.semantics.bindings import bound_sources
from udfx.semantics.formatting import fixed_format_exact
from udfx.semantics.ranges import integer_bounds
from udfx.testing import check_function
from udfx.types import DATE, FLOAT, INT, INT64_MAX, INT64_MIN


def days() -> Call:
    difference = apply("date_difference", Col("left", DATE), Col("right", DATE))
    return apply("__floordiv__", difference, lit(86400000000))


def formatted(integer: Node, factor: float, precision: Node) -> Call:
    return apply("format_fixed", apply("__mul__", integer, lit(factor)), precision)


@pytest.mark.parametrize(
    ("factor", "digits", "expected"),
    [
        (0.1, 2, True),
        (-0.1, 2, True),
        (0.0, 22, True),
        (-0.0, 22, True),
        (0.125, 0, True),
        (0.125, 2, False),
        (0.01, 1, False),
        (1.0, 22, False),
        (1e308, 2, False),
        (float("nan"), 2, False),
        (float("inf"), 2, False),
        (0.1, -1, False),
        (0.1, 23, False),
    ],
)
def test_bounded_integer_fixed_format_proof(factor: float, digits: int, expected: bool) -> None:
    assert fixed_format_exact(formatted(days(), factor, lit(digits))) is expected


@pytest.mark.parametrize("integer", [Col("value", INT), Lit(2**53 + 1, INT)])
def test_unbounded_or_inexact_integer_keeps_guard(integer: Node) -> None:
    assert not fixed_format_exact(formatted(integer, 0.1, lit(2)))


@pytest.mark.parametrize("integer", [-(2**53), 2**53])
def test_exact_integer_conversion_endpoint(integer: int) -> None:
    assert fixed_format_exact(formatted(lit(integer), 2.0**-20, lit(0)))


def test_dynamic_precision_and_float_input_keep_guard() -> None:
    assert not fixed_format_exact(formatted(days(), 0.1, Col("digits", INT)))
    assert not fixed_format_exact(apply("format_fixed", Col("value", FLOAT), lit(2)))


def test_date_bounds_and_bound_sources() -> None:
    assert integer_bounds(days(), INT64_MAX) == (-3652058, 3652058)
    integer, product, precision = (
        Element("integer", INT),
        Element("product", FLOAT),
        Element("precision", INT),
    )
    with bound_sources(
        {
            integer: days(),
            product: apply("__mul__", lit(-0.1), integer),
            precision: lit(2),
        }
    ):
        assert fixed_format_exact(apply("format_fixed", product, precision))


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(("factor", "digits"), [(0.1, 2), (-0.1, 2), (-0.0, 22), (0.125, 0)])
def test_proven_format_removes_only_redundant_guard(
    backend: Backend[Any], factor: float, digits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = formatted(days(), factor, lit(digits))
    schema = {"left": DATE, "right": DATE}
    dates = [date.min, date(1970, 1, 1), date.max]
    rows = [(left, right) for left in dates for right in dates]
    lowered = backend.lower(value)
    assert lowered.exact is None
    expected = [format((left - right).days * factor, f".{digits}f") for left, right in rows]
    assert backend.evaluate([lowered.expr], schema, rows) == [expected]
    options = backend.rules.options(value.op.key)
    monkeypatch.setitem(
        backend.rules._options,  # pyright: ignore[reportPrivateUsage]
        value.op.key,
        replace(options, facts={"exact": lambda _: False}),
    )
    guarded = backend.lower(value)
    assert guarded.exact is not None
    assert backend.evaluate([backend.exact_expression(guarded)], schema, rows) == [[True] * 9]


@pytest.mark.parametrize("backend", backend_params())
def test_proven_format_preserves_nullable_date_faults(backend: Backend[Any]) -> None:
    def result(left: date, right: date) -> str:
        try:
            return f"{(left - right).days * 0.1:.2f}"
        except TypeError:
            return "missing"

    rows = [(date.min, date.max), (date.max, date.min), (None, date.min), (date.max, None)]
    schema = {"left": DATE.or_none(), "right": DATE.or_none()}
    assert check_function(result, schema, backend, rows) == []
    assert backend.lower_program(trace(result, schema)).exact is None


@pytest.mark.parametrize("backend", backend_params())
def test_proven_format_preserves_upstream_placeholder_guard(backend: Backend[Any]) -> None:
    integer = apply("__sub__", Lit(INT64_MIN - 1, INT), Lit(INT64_MIN, INT))
    value = formatted(integer, 0.1, lit(2))
    assert fixed_format_exact(value)
    program = Program(lambda marker: "-0.10", {"marker": INT}, value)
    lowered = backend.lower_program(program)
    assert backend.evaluate([backend.exact_expression(lowered)], program.schema, [(0,)]) == [
        [False]
    ]
    assert backend.evaluate([backend.build(program)], program.schema, [(0,)]) == [["-0.10"]]


@pytest.mark.parametrize("backend", backend_params())
def test_proven_format_keeps_safe_native_placeholder(backend: Backend[Any]) -> None:
    value = Col("value", FLOAT)
    node = apply("format_fixed", value, lit(2))
    rule = backend.rules.get(node.op.key)
    assert rule is not None
    lowered = rule(backend.column(value), 2, exact=True)
    assert isinstance(lowered, Lowered)
    assert lowered.exact is None
    rows = [(1e308,), (-1e308,), (float("nan"),), (float("inf"),)]
    assert len(backend.evaluate([lowered.expr], {"value": FLOAT}, rows)[0]) == len(rows)
