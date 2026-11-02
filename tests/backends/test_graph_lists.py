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
