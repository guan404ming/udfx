from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import pytest
from conftest import backend_params, backends
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from udfx.backends.base import Backend, Row
from udfx.ops import Op, all_ops, resolve
from udfx.semantics.scalars import numpy_scalar
from udfx.testing import batches, values
from udfx.testing.batching import check_batched
from udfx.testing.equivalence import check_op_batches
from udfx.testing.strategies import (
    constant_values,
    extra_traps,
)
from udfx.types import INT, Kind, Type

CASES = [
    pytest.param(backend, op, id=f"{label}-{op}", marks=pytest.mark.xdist_group(label))
    for label, backend in backends()
    for op in all_ops()
]


@pytest.mark.parametrize("backend", backend_params())
def test_every_op_has_a_rule(backend: Backend[Any]) -> None:
    missing = {op.key for op in all_ops()} - backend.rules.keys()
    assert not missing


@pytest.mark.parametrize(("backend", "op"), CASES)
def test_op_matches_python(backend: Backend[Any], op: Op) -> None:
    if isinstance(op.reference, partial) and op.reference.func is numpy_scalar:
        pytest.importorskip("numpy")
    schema = {f"arg{index}": param for index, param in enumerate(op.parameters)}

    def draw(data: st.DataObject) -> tuple[list[Row], dict[int, object]]:
        rows = data.draw(batches(schema))
        neutral = tuple(
            1.0 if parameter.kind is Kind.FLOAT else 1 if parameter.kind is Kind.INT else value
            for parameter, value in zip(op.parameters, rows[0], strict=True)
        )
        for index, parameter in enumerate(op.parameters):
            for value in extra_traps(parameter):
                rows.append((*neutral[:index], value, *neutral[index + 1 :]))
        constant = data.draw(st.sampled_from([None, *range(len(op.parameters))]))
        constants = (
            {} if constant is None else {constant: data.draw(values(op.parameters[constant]))}
        )
        literals = backend.rules.options(op.key).literals
        constants.update(
            (index, data.draw(constant_values(op.parameters[index]))) for index in literals
        )
        return rows, constants

    def check(examples: Sequence[tuple[list[Row], dict[int, object]]]) -> None:
        assert check_op_batches(op, backend, examples) == []

    check_batched(draw, check)


def test_batching_keeps_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("udfx.testing.batching.getrandbits", lambda _: 0)
    drawn: list[list[int]] = []
    checked: list[Sequence[list[int]]] = []

    def draw(data: st.DataObject) -> list[int]:
        return data.draw(st.lists(st.integers(), min_size=1, max_size=40))

    @seed(0)
    @given(st.data())
    def unbatched(data: st.DataObject) -> None:
        drawn.append(draw(data))

    unbatched()
    check_batched(draw, checked.append)
    assert checked == [drawn]
    assert len(drawn) == settings().max_examples


def test_batching_shrinks_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("udfx.testing.batching.getrandbits", lambda _: 0)

    def draw(data: st.DataObject) -> list[int]:
        return data.draw(st.lists(st.integers(), min_size=1, max_size=40))

    def check(examples: Sequence[list[int]]) -> None:
        for rows in examples:
            assert all(value < 100 for value in rows), rows

    with pytest.raises(AssertionError) as error:
        check_batched(draw, check)
    assert str(error.value).splitlines()[0] == "[100]"


@pytest.mark.parametrize("backend", backend_params())
def test_op_batches_keep_constants(backend: Backend[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    op = resolve("__floordiv__", (INT, INT))
    assert op is not None
    evaluate = backend.evaluate
    calls = 0
    checked: list[Row] = []

    def count(
        exprs: Sequence[Any], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        nonlocal calls
        calls += 1
        checked.extend(rows)
        return evaluate(exprs, schema, rows)

    monkeypatch.setattr(backend, "evaluate", count)
    assert (
        check_op_batches(
            op,
            backend,
            [([(4, 1), (7, 1)], {1: 2}), ([(9, 1)], {1: 0}), ([(11, 1)], {1: 2})],
        )
        == []
    )
    assert calls == 1
    assert {0, 2, 4, 7, 9, 11} <= {value for row in checked for value in row}
