from __future__ import annotations

from typing import Any

import pytest
from conftest import backend_params
from hypothesis import given
from hypothesis import strategies as st

from udfx import compile
from udfx.backends.base import Backend
from udfx.testing import batches, check_function

from .cases import CASES, Case

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_case(case: Case, backend: Backend[Any]) -> None:
    translation = compile(case.function, case.schema, backend)
    if case.expect == "native":
        assert translation.ok, translation.unsupported
    if not translation.ok:
        return

    @given(st.data())
    def check(data: st.DataObject) -> None:
        rows = data.draw(batches(case.schema))
        assert check_function(case.function, case.schema, backend, rows) == []

    check()
