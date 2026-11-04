import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_consecutive_transforms_use_one_pass(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        return [value + "!" for value in stripped]

    schema = {"values": list_of(STR)}
    with patch.object(backend, "list_transform", wraps=backend.list_transform) as transform:
        expr = backend.build(trace(mapped, schema))
    assert transform.call_count == 1
    rows = [([],), ([" a_b ", "\u2003中文_😀\u2003", ""],)]
    assert backend.evaluate([expr], schema, rows) == [[mapped(values) for (values,) in rows]]


@pytest.mark.parametrize("backend", backend_params())
def test_transform_composition_keeps_guards(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[str]:
        upper = [value.upper() for value in values]
        return [value + "!" for value in upper]

    rows = [([],), (["a", "ß", "ﬃ", "\u0131", "中文", "😀"],)]
    assert check_function(mapped, {"values": list_of(STR)}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_transform_composition_keeps_reused_values(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        return [value + value for value in stripped]

    schema = {"values": list_of(STR)}
    with patch.object(backend, "list_transform", wraps=backend.list_transform) as transform:
        expr = backend.build(trace(mapped, schema))
    assert transform.call_count == 2
    rows = [([],), ([" a ", "", "中文"],)]
    assert backend.evaluate([expr], schema, rows) == [[mapped(values) for (values,) in rows]]


@pytest.mark.parametrize("backend", backend_params())
def test_transform_composition_keeps_exception_order(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[str]:
        try:
            numbers = [int(value) for value in values]
            return [str(100 // value) for value in numbers]
        except ValueError:
            return ["value"]
        except ZeroDivisionError:
            return ["zero"]

    rows = [([],), (["0", "invalid"],), (["invalid", "0"],), (["2", "4"],), (["0"],)]
    assert check_function(mapped, {"values": list_of(STR)}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda values: len([value.strip() for value in values]),
        lambda values: bool([value.strip() for value in values]),
    ],
)
def test_cardinality_skips_unused_transform(
    backend: Backend[Any], function: Callable[[list[str]], object]
) -> None:
    schema = {"values": list_of(STR)}
    with patch.object(backend, "list_transform", wraps=backend.list_transform) as transform:
        expr = backend.build(trace(function, schema))
    transform.assert_not_called()
    rows = [([],), ([" a ", "", "中文"],)]
    assert backend.evaluate([expr], schema, rows) == [[function(values) for (values,) in rows]]


@pytest.mark.parametrize("backend", backend_params())
def test_cardinality_preserves_transform_errors(backend: Backend[Any]) -> None:
    def counted(values: list[str]) -> int:
        try:
            return len([int(value) for value in values])
        except ValueError:
            return -1

    rows = [([],), (["1", "2"],), (["1", "invalid"],), ([str(2**80)],)]
    assert check_function(counted, {"values": list_of(STR)}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_transform_composition_keeps_expanding_rules(backend: Backend[Any]) -> None:
    def encoded(values: list[str]) -> str:
        stripped = [value.strip() for value in values]
        return json.dumps(stripped, ensure_ascii=False)

    schema = {"values": list_of(STR)}
    with patch.object(backend, "list_transform", wraps=backend.list_transform) as transform:
        expr = backend.build(trace(encoded, schema))
    assert transform.call_count == (1 if backend.name == "spark" else 2)
    rows = [([],), ([" a ", "\n", "中文", '"'],)]
    assert backend.evaluate([expr], schema, rows) == [[encoded(values) for (values,) in rows]]


@pytest.mark.parametrize("backend", backend_params())
def test_transform_composition_changes_element_type(backend: Backend[Any]) -> None:
    def lengths(values: list[str]) -> list[int]:
        stripped = [value.strip() for value in values]
        return [len(value) for value in stripped]

    schema = {"values": list_of(STR)}
    with patch.object(backend, "list_transform", wraps=backend.list_transform) as transform:
        expr = backend.build(trace(lengths, schema))
    assert transform.call_count == 1
    rows = [([],), ([" a ", "", "中文", "😀"],)]
    assert backend.evaluate([expr], schema, rows) == [[lengths(values) for (values,) in rows]]
