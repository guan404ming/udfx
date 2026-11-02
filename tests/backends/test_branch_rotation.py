import json
import math
from itertools import product
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.frontend.core import branches
from udfx.frontend.core.branches import factor_branches
from udfx.ir import Col, Guarded, If, digest, lit
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("shape", range(4))
def test_shared_branch_keeps_short_circuit_guards(backend: Backend[Any], shape: int) -> None:
    schema = {
        "first": BOOL,
        "first_valid": BOOL.or_none(),
        "second": BOOL,
        "second_valid": BOOL.or_none(),
        "third": BOOL,
    }
    first = Guarded(Col("first", BOOL), Col("first_valid", BOOL.or_none()))
    second = Guarded(Col("second", BOOL), Col("second_valid", BOOL.or_none()))
    shared = If(Col("third", BOOL), lit(-0.0), lit(1.0), FLOAT)
    selected = lit(0.0)
    nodes = (
        If(first, If(second, selected, shared, FLOAT), shared, FLOAT),
        If(first, If(second, shared, selected, FLOAT), shared, FLOAT),
        If(first, shared, If(second, selected, shared, FLOAT), FLOAT),
        If(first, shared, If(second, shared, selected, FLOAT), FLOAT),
    )
    original = nodes[shape]
    (factored,) = factor_branches((original,))
    assert isinstance(factored, If) and isinstance(factored.condition, If)
    assert digest(original) != digest(factored)
    rows = list(
        product(
            (False, True), (False, True, None), (False, True), (False, True, None), (False, True)
        )
    )
    lowered = [backend.lower(node) for node in (original, factored)]
    masks = backend.evaluate([backend.exact_expression(part) for part in lowered], schema, rows)
    assert masks[0] == masks[1]
    values = backend.evaluate([part.expr for part in lowered], schema, rows)
    for before, after in zip(*values, strict=True):
        assert before == after
        if before == 0:
            assert math.copysign(1, before) == math.copysign(1, after)


@pytest.mark.parametrize("backend", backend_params())
def test_shared_branch_keeps_conversion_faults(backend: Backend[Any]) -> None:
    def converted(flag: bool, text: str, default: int) -> int:
        if flag:
            try:
                return int(text)
            except ValueError:
                return default + 1
        return default + 1

    schema = {"flag": BOOL, "text": STR.or_none(), "default": INT}
    rows = list(product((False, True), (None, "", "x", "12", "٣٢"), (-1, 0, 5)))
    assert check_function(converted, schema, backend, rows) == []
    valid = [row for row in rows if row[1] is not None]
    programs = []
    with patch.object(branches, "shared_branch", side_effect=lambda node: node):
        programs.append(trace(converted, schema))
    programs.append(trace(converted, schema))
    assert digest(programs[0].value) != digest(programs[1].value)
    masks = []
    for program in programs:
        with capture_lowering() as details:
            backend.build(program)
        exact = details[-1].exact
        masks.append(
            [True] * len(valid)
            if exact is None
            else backend.evaluate([backend.predicate(exact)], schema, valid)[0]
        )
    assert masks[0] == masks[1]


@pytest.mark.parametrize("backend", backend_params())
def test_rotated_conditions_preserve_arithmetic_bounds(backend: Backend[Any]) -> None:
    def bounded(flag: bool, value: int, default: int) -> int:
        if flag and 0 <= value < 10:
            return value + 1
        return default

    schema = {"flag": BOOL, "value": INT, "default": INT}
    rows = [
        (flag, value, -2) for flag, value in product((False, True), (-(2**63), -1, 0, 9, 2**63 - 1))
    ]
    program = trace(bounded, schema)
    assert isinstance(program.value, If) and isinstance(program.value.condition, If)
    lowered = backend.lower_program(program)
    assert lowered.exact is None
    assert check_function(bounded, schema, backend, rows) == []


def test_json_literal_defaults_keep_native_scalar_failures() -> None:
    def selected(flag: bool, text: str) -> bool:
        if flag:
            try:
                return bool(json.dumps([word.lower() for word in json.loads(text)]))
            except (ValueError, TypeError, AttributeError):
                return False
        return False

    backend = PolarsBackend()
    schema = {"flag": BOOL, "text": STR}
    rows = list(product((False, True), ("true", "false", "null", "1", "NaN", "Infinity", "1e1000")))
    with capture_lowering() as details:
        expr = backend.build(trace(selected, schema))
    assert backend.evaluate([expr], schema, rows) == [[False] * len(rows)]
    exact = details[-1].exact
    assert isinstance(exact, pl.Expr)
    assert backend.evaluate([backend.predicate(exact)], schema, rows) == [[True] * len(rows)]
