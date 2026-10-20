from typing import Any

import pytest
from conftest import backend_params
from hypothesis import given
from hypothesis import strategies as st

from udfx.backends import installed
from udfx.backends.base import Backend
from udfx.ops import Op, all_ops
from udfx.testing import batches, check_op, values

BACKENDS: tuple[Backend[Any], ...] = installed()
CASES = [
    pytest.param(
        backend, op, id=f"{backend.name}-{op}", marks=pytest.mark.xdist_group(backend.name)
    )
    for backend in BACKENDS
    for op in all_ops()
]


@pytest.mark.parametrize("backend", backend_params())
def test_every_op_has_a_rule(backend: Backend[Any]) -> None:
    missing = {op.key for op in all_ops()} - backend.rules.keys()
    assert not missing


@pytest.mark.parametrize(("backend", "op"), CASES)
@given(data=st.data())
def test_op_matches_python(backend: Backend[Any], op: Op, data: st.DataObject) -> None:
    schema = {f"arg{index}": param for index, param in enumerate(op.params)}
    rows = data.draw(batches(schema))
    constant = data.draw(st.sampled_from([None, *range(len(op.params))]))
    constants = {} if constant is None else {constant: data.draw(values(op.params[constant]))}
    assert check_op(op, backend, rows, constants) == []
