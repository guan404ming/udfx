from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from udfx.backends import installed
from udfx.backends.base import Backend
from udfx.ops import Op, all_ops
from udfx.testing import batches, check_op

BACKENDS: tuple[Backend[Any], ...] = installed()
CASES = [(backend, op) for backend in BACKENDS for op in all_ops()]


def case_id(case: object) -> str:
    return case.name if isinstance(case, Backend) else str(case)


@pytest.mark.parametrize("backend", BACKENDS, ids=case_id)
def test_every_op_has_a_rule(backend: Backend[Any]) -> None:
    missing = {op.key for op in all_ops()} - backend.rules.keys()
    assert not missing


@pytest.mark.parametrize(("backend", "op"), CASES, ids=case_id)
@given(data=st.data())
def test_op_matches_python(backend: Backend[Any], op: Op, data: st.DataObject) -> None:
    schema = {f"arg{index}": param for index, param in enumerate(op.params)}
    rows = data.draw(batches(schema))
    assert check_op(op, backend, rows) == []
