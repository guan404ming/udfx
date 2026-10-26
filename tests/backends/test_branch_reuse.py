from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, Guarded, If, lit
from udfx.ops import apply
from udfx.semantics.assumptions import assuming
from udfx.testing import check_function
from udfx.types import BOOL, INT, STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_branch_reuses_condition_guard(backend: Backend[Any]) -> None:
    text = Col("text", STR)
    valid = Col("valid", BOOL)
    guarded = Guarded(text, valid)
    node = If(apply("bool", guarded), apply("len", guarded), lit(0), INT)
    lowered = backend.lower(node)
    assert str(lowered.exact).count("valid") == 1
    assert backend.evaluate(
        [lowered.exact], {"text": STR, "valid": BOOL}, [("x", True), ("x", False)]
    ) == [[True, False]]
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
    assert backend.evaluate([lowered.exact], schema, rows) == [[False, False, True]]


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
    assert backend.evaluate([guard], {"text": STR}, rows) == [[True, True, False]]


@pytest.mark.parametrize("backend", backend_params())
def test_literal_selection_uses_one_case_chain(backend: Backend[Any]) -> None:
    def lookup(value: str) -> float:
        return {"red": 1.5, "green": 2.5, "blue": 3.5}.get(value, 0.0)

    program = trace(lookup, {"value": STR})
    with patch.object(backend, "select_many", wraps=backend.select_many) as select:
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
    with patch.object(backend, "bind", wraps=backend.bind) as bind:
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
    with patch.object(backend, "native_raise", wraps=backend.native_raise) as raise_error:
        expr = backend.build(program)
    raise_error.assert_not_called()
    rows = [("",), ("::",), ("a::b",), ("a::::b",), (":::::",), ("😀::c",)]
    assert backend.evaluate([expr], program.schema, rows) == [[joined(text) for (text,) in rows]]


@pytest.mark.parametrize("backend", [value for value in backend_params() if value.id == "spark"])
def test_result_overflow_stays_inside_selected_branch(backend: Backend[Any]) -> None:
    def parsed(text: str) -> int:
        try:
            return int(text)
        except ValueError:
            return -1

    schema = {"text": STR}
    expr = backend.build(trace(parsed, schema))
    assert str(expr).lower().count("rlike(") == 1
    rows = [("12",), ("invalid",), ("٣٢",), ("9223372036854775808",)]
    assert check_function(parsed, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_predicate_selection_uses_one_case_chain(backend: Backend[Any]) -> None:
    def classify(text: str | None) -> str:
        if text is None:
            return "unknown"
        if "first" in text.lower():
            return "first"
        if "second" in text.lower():
            return "second"
        return "other"

    schema = {"text": STR.or_none()}
    with patch.object(backend, "select_many", wraps=backend.select_many) as select:
        backend.build(trace(classify, schema))
    select.assert_called_once()
    rows = [(None,), ("FIRST SECOND",), ("SECOND",), ("fİrst",), ("other",)]
    assert check_function(classify, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_equal_case_results_share_one_search(backend: Backend[Any]) -> None:
    def classify(text: str | None) -> str:
        if text is None:
            return "unknown"
        if "first" in text.lower():
            return "first"
        if "second" in text.lower() or "third" in text.lower():
            return "other"
        return "missing"

    schema = {"text": STR.or_none()}
    with patch.object(backend, "select_many", wraps=backend.select_many) as select:
        backend.build(trace(classify, schema))
    assert len(select.call_args.args[0]) == 3
    values = [None, "", "THIRD FIRST", "second", "THIRD", "thİrd", "missing"]
    assert check_function(classify, schema, backend, [(value,) for value in values]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_equal_case_results_keep_signed_zero(backend: Backend[Any]) -> None:
    def classify(text: str) -> float:
        if text == "positive":
            return 0.0
        if text == "negative":
            return -0.0
        return 1.0

    assert (
        check_function(classify, {"text": STR}, backend, [("positive",), ("negative",), ("other",)])
        == []
    )
