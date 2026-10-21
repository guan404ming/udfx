from typing import Any, cast

import pytest
from conftest import backend_params, backends
from hypothesis import given
from hypothesis import strategies as st

from udfx.backends.base import Backend
from udfx.ops import Op, all_ops
from udfx.semantics.text import PREDICATES
from udfx.testing import batches, check_op, values
from udfx.testing.strategies import (
    PREDICATE_TRAPS,
    REGEX_PATTERNS,
    SLICE_TRAPS,
    STRING_TRAPS,
    regex_constants,
)
from udfx.types import Kind

CASES = [
    pytest.param(backend, op, id=f"{label}-{op}", marks=pytest.mark.xdist_group(backend.name))
    for label, backend in backends()
    for op in all_ops()
]


@pytest.mark.parametrize("backend", backend_params())
def test_every_op_has_a_rule(backend: Backend[Any]) -> None:
    missing = {op.key for op in all_ops()} - backend.rules.keys()
    assert not missing


@pytest.mark.parametrize(("backend", "op"), CASES)
@given(data=st.data())
def test_op_matches_python(backend: Backend[Any], op: Op, data: st.DataObject) -> None:
    if op.name.startswith("numpy."):
        pytest.importorskip("numpy")
    schema = {f"arg{index}": param for index, param in enumerate(op.parameters)}
    rows = data.draw(batches(schema))
    if op.name in {f".{method}" for method in PREDICATES}:
        rows.extend((value,) for value in PREDICATE_TRAPS)
    constant = data.draw(st.sampled_from([None, *range(len(op.parameters))]))
    constants = {} if constant is None else {constant: data.draw(values(op.parameters[constant]))}
    if op.name.startswith("re."):
        constants.update(data.draw(regex_constants(op.name)))
    if op.name == "slice":
        bound_values = st.one_of(values(op.parameters[1]), st.sampled_from(SLICE_TRAPS))
        bounds = data.draw(st.sets(st.sampled_from([1, 2])))
        constants.update((index, data.draw(bound_values)) for index in bounds)
        if len(op.parameters) == 4:
            constants[3] = data.draw(bound_values)
    assert check_op(op, backend, rows, constants) == []


@pytest.mark.parametrize(
    ("backend", "op"), [case for case in CASES if cast("Op", case.values[1]).name.startswith("re.")]
)
def test_regex_traps(backend: Backend[Any], op: Op) -> None:
    rows = [
        (text, *("" if parameter.kind is Kind.STR else 0 for parameter in op.parameters[1:]))
        for text in STRING_TRAPS
    ]
    for pattern in REGEX_PATTERNS:
        constants: dict[int, object] = {1: pattern, 2: 0}
        if op.name == "re.sub":
            constants.update({3: r"\g<0>-$1", 4: 0})
            if "(" in pattern:
                constants[3] = r"\g<1>0"
        if op.name == "re.split":
            constants[3] = 0
        assert check_op(op, backend, rows, constants) == [], pattern
