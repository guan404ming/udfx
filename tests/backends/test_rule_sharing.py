import math
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.program_lowering import choosing_graph
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import FLOAT


@pytest.mark.parametrize("backend", backend_params())
def test_repeated_rule_argument_binds_once(backend: Backend[Any]) -> None:
    def rounded(value: float) -> float:
        return round(math.sqrt(value * value + 1.0) * 3.0, 2)

    schema = {"value": FLOAT}
    with patch.object(backend, "bind", wraps=backend.bind) as bind, choosing_graph(False):
        backend.build(trace(rounded, schema))
    assert bind.called == type(backend).binds
    rows = [(0.0,), (1.5,), (-2.25,), (1e300,), (float("nan"),), (float("inf"),)]
    assert check_function(rounded, schema, backend, rows) == []
