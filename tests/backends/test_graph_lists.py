from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.backends.program_lowering import choosing_graph
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


def shout(item: str) -> str:
    word = item.strip("'")
    return word.upper() if word.endswith("s") else word + word


def shouted(text: str) -> str:
    return " ".join([shout(item) for item in text.split()])


@pytest.mark.parametrize("backend", backend_params())
def test_unguarded_graph_comprehension_skips_records(backend: Backend[Any]) -> None:
    if not backend.supports_graphs():
        pytest.skip("graph lowering unavailable")

    def joined(text: str) -> str:
        cleaned = text.strip()
        return cleaned + str(len([item for item in cleaned if item in ["a", "e"]]))

    schema = {"text": STR}
    program = trace(joined, schema)
    with patch.object(backend, "record", wraps=backend.record) as record, choosing_graph(True):
        backend.build(program)
    assert record.call_count == 2
    rows = [("",), (" banana ",), ("été",), ("xyz",)]
    assert check_function(joined, schema, backend, rows) == []


@pytest.mark.xdist_group("polars")
def test_guarded_graph_comprehension_lowers_body_once() -> None:
    backend = PolarsBackend()
    schema = {"text": STR}
    with choosing_graph(True):
        lowered = backend.lower_program(trace(shouted, schema))
    packed = lowered.consume(
        lambda part: backend.record([part.expr, backend.exact_expression(part)])
    )
    assert str(packed).count("strip") == 2
    rows = [("",), ("'cats' dog",), ("straße s",), ("ǅs x",)]
    assert check_function(shouted, schema, backend, rows) == []
