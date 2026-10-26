from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.explain import render
from udfx.frontend import trace
from udfx.ir import Guarded
from udfx.testing import check_function
from udfx.types import INT, STR, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [lambda items: items * 3, lambda items: 3 * items])
def test_constant_list_repeat_uses_concatenation(
    backend: Backend[Any], function: Callable[..., object]
) -> None:
    schema = {"items": list_of(STR)}
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    assert not fallback_in_plan(translation.expr, backend.name)
    rows = [([],), (["a", "😀"],), (["word"] * 10001,)]
    assert check_function(function, schema, backend, rows) == []
    assert check_function(function, {"items": list_of(STR).or_none()}, backend, [(None,)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_empty_repeat_keeps_operand_faults(backend: Backend[Any]) -> None:
    def empty(text: str) -> list[int]:
        return [int(text)] * 0

    assert check_function(empty, {"text": STR}, backend, [("1",), ("invalid",)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_partitioned_sum_keeps_short_lists(backend: Backend[Any]) -> None:
    def total(values: list[int]) -> int:
        return values[0] + sum(values[1:-1]) + values[-1]

    schema = {"values": list_of(INT).or_none()}
    program = trace(total, schema)
    assert isinstance(program.value, Guarded)
    rows = [(None,), ([],), ([1],), ([1, 2],), ([2**63 - 1, 1, -(2**63 - 1)],)]
    assert check_function(total, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(("start", "stop"), [(0, None), (1, None), (0, -1), (1, -1), (3, -2)])
def test_joined_word_slice(backend: Backend[Any], start: int, stop: int | None) -> None:
    def joined(text: str) -> str:
        return " ".join(text.split()[start:stop])

    schema = {"text": STR.or_none()}
    rows = [(None,), ("",), (" a ",), (" a\tb\nc ",), ("a\x1cb\u2003c\u3000d\u0085e",)]
    assert check_function(joined, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("separator", ["_", ".", "\\", "\n", "😀", "]"])
def test_joined_split_prefix_keeps_empty_fields(backend: Backend[Any], separator: str) -> None:
    def joined(text: str) -> str:
        return separator.join(text.split(separator)[:3])

    schema = {"text": STR.or_none()}
    program = trace(joined, schema)
    assert program.result == STR
    assert "re.match(" in render(program.value)
    assert ".split(" not in render(program.value)
    rows = [
        (None,),
        ("",),
        (separator,),
        (separator * 2 + "a" + separator,),
        (separator.join(["a", "\nb", "c", ""]),),
    ]
    assert check_function(joined, schema, backend, rows) == []
