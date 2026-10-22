from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from conftest import backend_params
from hypothesis import strategies as st

from udfx import compile
from udfx.backends.base import Backend, Row
from udfx.testing import Raised, batches
from udfx.testing.batching import check_batched
from udfx.testing.equivalence import check_translation_batches
from udfx.types import INT, Type

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

    def draw(data: st.DataObject) -> list[Row]:
        return data.draw(batches(case.schema))

    def check(examples: Sequence[list[Row]]) -> None:
        assert check_translation_batches(translation, backend, examples) == []

    check_batched(draw, check)


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator


@pytest.mark.parametrize("backend", backend_params())
def test_batches_keep_exception_checks(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    translation = compile(divide, {"numerator": INT, "denominator": INT}, backend)
    evaluate = backend.evaluate
    checked: list[Sequence[Row]] = []
    wrong_kind = False

    def record(
        exprs: Sequence[Any], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        checked.append(rows)
        if wrong_kind and rows == [(5, 0)]:
            raise ValueError
        return evaluate(exprs, schema, rows)

    monkeypatch.setattr(backend, "evaluate", record)
    examples = [[(2, 1), (3, 0)], [(4, 1), (5, 0)], [(2, 1), (3, 0)]]
    assert check_translation_batches(translation, backend, examples) == []
    assert checked == [
        [(2, 1), (4, 1), (2, 1)],
        [(3, 0)],
        [(5, 0)],
        [row for rows in examples for row in rows],
    ]
    wrong_kind = True
    mismatches = check_translation_batches(translation, backend, examples)
    assert len(mismatches) == 1
    assert mismatches[0].row == (5, 0)
    assert mismatches[0].expected == Raised(ZeroDivisionError)
