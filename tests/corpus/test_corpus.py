from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from udfx import compile
from udfx.backends.polars import PolarsBackend
from udfx.testing import batches, check_function

from .cases import CASES, Case

pytestmark = pytest.mark.corpus

POLARS = PolarsBackend()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_case(case: Case) -> None:
    translation = compile(case.function, case.schema, POLARS)
    if case.expect == "native":
        assert translation.ok, translation.unsupported
    if not translation.ok:
        return

    @given(st.data())
    def check(data: st.DataObject) -> None:
        rows = data.draw(batches(case.schema))
        assert check_function(case.function, case.schema, POLARS, rows) == []

    check()
