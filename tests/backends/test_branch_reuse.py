from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.hoisting import repeated_condition
from udfx.backends.program_lowering import choosing_graph
from udfx.frontend import trace
from udfx.ir import Col, Guarded, If, lit
from udfx.ops import apply
from udfx.program import Program
from udfx.semantics.assumptions import assuming
from udfx.testing import check_function
from udfx.types import BOOL, INT, STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_branch_reuses_list_cardinality(backend: Backend[Any]) -> None:
    def selected(words: list[str]) -> int:
        values = sorted(words + words)
        return len(values) + len(values) if len(values) > 2 else len(values)

    schema = {"words": list_of(STR)}
    program = trace(selected, schema)
    assert isinstance(program.value, If)
    assert repeated_condition(program.value, {}, backend.hoisted) is not None
    rows = [([],), (["a"],), (["a", "b"],), (["é", "中", ""],)]
    assert check_function(selected, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_branch_reuses_condition_guard(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    valid = Col("valid", BOOL)
    guarded = Guarded(text, valid)
    node = If(apply("bool", guarded), apply("len", guarded), lit(0), INT)
    lowered = backend.lower(node)
    assert str(lowered.exact).count("valid") == 1
    exact = backend.predicate(lowered.exact)
    assert backend.evaluate([exact], {"text": STR, "valid": BOOL}, [("x", True), ("x", False)]) == [
        [True, False]
    ]
    assert backend.lower(guarded).exact is not None


@pytest.mark.parametrize("backend", backend_params())
def test_unevaluated_branch_does_not_prove_guard(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    valid = Col("valid", BOOL)
    flag = Col("flag", BOOL)
    guarded = Guarded(text, valid)
    condition = apply("bool", If(flag, guarded, text, STR))
    lowered = backend.lower(If(condition, apply("len", guarded), lit(0), INT))
    schema = {"text": STR, "valid": BOOL, "flag": BOOL}
    rows = [("x", False, False), ("x", False, True), ("x", True, False)]
    flags = backend.evaluate([backend.predicate(lowered.exact)], schema, rows)[0]
    assert flags == [False, False, True]


@pytest.mark.parametrize("backend", backend_params())
def test_validated_integer_keeps_overflow_guard(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    with assuming(apply("int_parses", text), True):
        parsed = backend.lower(apply("int", text))
    guard = backend.exact(parsed)
    assert guard is not None
    assert all(
        name not in str(guard).lower()
        for name in ("regexp_matches", "regexp_like", "rlike", "str.contains")
    )
    rows = [("12",), ("٣٢",), ("9223372036854775808",)]
    flags = backend.evaluate([backend.predicate(guard)], {"text": STR}, rows)[0]
    assert flags == [True, True, False]


@pytest.mark.parametrize("backend", backend_params())
def test_literal_selection_uses_one_native_selection(backend: Backend[Any]) -> None:
    def lookup(value: str) -> float:
        return {"red": 1.5, "green": 2.5, "blue": 3.5}.get(value, 0.0)

    program = trace(lookup, {"value": STR})
    name = "select_value" if backend.name in {"duckdb", "datafusion"} else "select_many"
    with patch.object(backend, name, wraps=getattr(backend, name)) as select:
        expr = backend.build(program)
    select.assert_called_once()
    rows = [("red",), ("green",), ("blue",), ("unknown",)]
    assert backend.evaluate([expr], program.schema, rows) == [[lookup(value) for (value,) in rows]]


@pytest.mark.parametrize("backend", backend_params())
def test_branch_with_mutable_list_literal(backend: Backend[Any]) -> None:
    kind = list_of(STR)
    node = If(Col("flag", BOOL), lit(["x"], kind), lit([], kind), kind)
    lowered = backend.lower(node)
    assert backend.evaluate([lowered.expr], {"flag": BOOL}, [(True,), (False,)]) == [[["x"], []]]


@pytest.mark.parametrize("backend", backend_params())
def test_single_operation_reuse_does_not_bind(backend: Backend[Any]) -> None:
    def selected(text: str) -> str:
        stripped = text.strip()
        return stripped if stripped else "missing"

    program = trace(selected, {"text": STR})
    with patch.object(backend, "bind", wraps=backend.bind) as bind, choosing_graph(False):
        expr = backend.build(program)
    bind.assert_not_called()
    rows = [(" a ",), ("",), ("  ",)]
    assert backend.evaluate([expr], program.schema, rows) == [["a", "missing", "missing"]]


@pytest.mark.parametrize("backend", backend_params())
def test_split_count_proves_index_bounds(backend: Backend[Any]) -> None:
    def joined(text: str) -> str:
        if text.count("::") == 1:
            pieces = text.split("::")
            return pieces[0] + "/" + pieces[1]
        return text

    program = trace(joined, {"text": STR})
    with (
        patch.object(backend, "native_raise", wraps=backend.native_raise) as raise_error,
        choosing_graph(False),
    ):
        expr = backend.build(program)
    raise_error.assert_not_called()
    rows = [("",), ("::",), ("a::b",), ("a::::b",), (":::::",), ("😀::c",)]
    assert backend.evaluate([expr], program.schema, rows) == [[joined(text) for (text,) in rows]]


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"duckdb", "spark"}]
)
def test_result_overflow_stays_inside_selected_branch(backend: Backend[Any]) -> None:
    def parsed(text: str) -> int:
        try:
            return int(text)
        except ValueError:
            return -1

    schema = {"text": STR}
    expr = backend.build(trace(parsed, schema))
    assert str(expr).lower().count("rlike(" if backend.name == "spark" else "regexp_matches(") == 1
    rows = [("12",), ("invalid",), ("٣٢",), ("9223372036854775808",)]
    assert check_function(parsed, schema, backend, rows) == []


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id == "datafusion"]
)
def test_result_fallback_stays_inside_selected_branch(backend: Backend[Any]) -> None:
    def parsed(text: str) -> int:
        try:
            return int(text)
        except ValueError:
            return -1

    schema = {"text": STR}
    program = trace(parsed, schema)
    with (
        capture_lowering() as details,
        patch.object(backend, "fallback", wraps=backend.fallback) as fallback,
    ):
        expr = backend.build(program)
    fallback.assert_called_once()
    assert "regexp_like" not in str(fallback.call_args.args[1])
    assert str(expr).count("regexp_like") == 1
    assert len(details) == 1
    rows = [("12",), ("invalid",), ("٣٢",), ("9223372036854775808",)]
    flags = backend.evaluate([backend.predicate(details[0].exact)], schema, rows)[0]
    assert flags == [True, True, True, False]
    assert check_function(parsed, schema, backend, rows) == []


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"datafusion", "duckdb"}]
)
def test_nested_branch_keeps_condition_guard_inside_branch(backend: Backend[Any]) -> None:
    schema = {"flag": BOOL, "value": BOOL, "valid": BOOL.or_none()}
    condition = Guarded(Col("value", BOOL), Col("valid", BOOL.or_none()))
    value = If(Col("flag", BOOL), If(condition, lit(10), lit(20), INT), lit(0), INT)
    program = Program(
        lambda flag, value, valid: (1 if value else 2) * (10 if valid else 1) if flag else 0,
        schema,
        value,
    )
    with (
        capture_lowering() as details,
        patch.object(backend, "fallback", wraps=backend.fallback) as fallback,
    ):
        expr = backend.build(program)
    fallback.assert_called_once()
    assert "flag" not in str(fallback.call_args.args[1])
    rows = [(False, True, False), (True, True, True), (True, False, False), (True, True, None)]
    assert backend.evaluate([expr], schema, rows) == [[0, 10, 2, 1]]
    assert len(details) == 1
    flags = backend.evaluate([backend.predicate(details[0].exact)], schema, rows)[0]
    assert flags == [True, True, False, None]
