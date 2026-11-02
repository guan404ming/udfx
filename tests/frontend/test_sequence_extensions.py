# ruff: noqa: PLW0108

from collections.abc import Callable
from typing import Any

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Call
from udfx.testing import check_function
from udfx.types import INT, STR, list_of


def first_position(values: list[str], item: str) -> int:
    try:
        return values.index(item)
    except ValueError:
        return -1


@pytest.mark.parametrize("backend", frontend_backend_params())
def test_list_index_exception_flow(backend: Backend[Any]) -> None:
    rows = [([], ""), (["x", "x", "y"], "x"), (["x", ""], ""), (["x"], "y")]
    assert (
        check_function(first_position, {"values": list_of(STR), "item": STR}, backend, rows) == []
    )


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: "*" * len(value),
        lambda value: len(value) * "*",
        lambda value: ("a" + value) * 3,
    ],
)
def test_string_repetition_forms(function: Callable[..., object], backend: Backend[Any]) -> None:
    assert (
        check_function(function, {"value": STR.or_none()}, backend, [("",), ("é😀",), (None,)])
        == []
    )


def test_literal_repetition_defers_allocation() -> None:
    program = trace(lambda: "x" * 2**63, {})
    assert isinstance(program.value, Call)
    assert program.result == STR


@pytest.mark.parametrize("backend", frontend_backend_params())
def test_negative_repetition_is_empty(backend: Backend[Any]) -> None:
    def function(value: int) -> str:
        return "ab" * (value - 2)

    assert check_function(function, {"value": INT}, backend, [(-1,), (0,), (2,), (4,)]) == []


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize(
    "function",
    [lambda value: "".join(sorted(value)), lambda value: any(value), lambda value: all(value)],
)
def test_string_iterable_functions(function: Callable[..., object], backend: Backend[Any]) -> None:
    rows = [("",), ("bAa",), ("\ue000\U00010000😀",), (None,)]
    assert check_function(function, {"value": STR.or_none()}, backend, rows) == []


def caught_index(values: list[object], item: object) -> int:
    try:
        return values.index(item)
    except ValueError:
        return -1
    except TypeError:
        return -2
    except AttributeError:
        return -3


@pytest.mark.parametrize("backend", frontend_backend_params())
def test_nullable_index_exception_order(backend: Backend[Any]) -> None:
    rows = [
        (["a"], None),
        ([], None),
        (None, None),
        (None, "missing"),
        (["a"], "missing"),
        (["a", "b", "a"], "a"),
    ]
    schema = {"values": list_of(STR).or_none(), "item": STR.or_none()}
    assert check_function(caught_index, schema, backend, rows) == []
