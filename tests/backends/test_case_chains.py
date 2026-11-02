from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.program_lowering import choosing_graph
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


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
    with (
        patch.object(backend, "select_many", wraps=backend.select_many) as select,
        choosing_graph(False),
    ):
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
