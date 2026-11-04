from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import FALSE, Call, If
from udfx.testing import check_function
from udfx.types import INT, STR, list_of


def selected(values: list[str | None]) -> str | None:
    try:
        return values[2]
    except IndexError:
        return None


def test_caught_index_error_uses_optional_item() -> None:
    program = trace(selected, {"values": list_of(STR.or_none())})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "optional_item"
    assert program.fault == FALSE


@pytest.mark.parametrize("limit", [1, 3])
def test_optional_item_keeps_different_length_conditions(limit: int) -> None:
    def guarded(values: list[str]) -> str | None:
        return values[2] if len(values) > limit else None

    program = trace(guarded, {"values": list_of(STR)})
    assert isinstance(program.value, If)


@pytest.mark.parametrize("backend", backend_params())
def test_caught_index_error_preserves_nulls_and_faults(backend: Backend[Any]) -> None:
    schema = {"values": list_of(STR.or_none()).or_none()}
    rows = [(None,), ([],), (["a"],), (["a", "b", None],), (["a", "b", ""],)]
    assert check_function(selected, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_optional_item_keeps_source_faults(backend: Backend[Any]) -> None:
    def divided(values: list[int]) -> int | None:
        try:
            return [10 // value for value in values][2]
        except IndexError:
            return None

    rows = [([],), ([1],), ([0],), ([1, 2, 5],), ([1, 2, 5, 0],)]
    assert check_function(divided, {"values": list_of(INT)}, backend, rows) == []
