from collections.abc import Callable, Iterable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Call, Guarded, If
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("reduction", [any, all])
def test_generator_short_circuit_guards(
    backend: Backend[Any], reduction: Callable[[Iterable[bool]], bool]
) -> None:
    def reduced(text: str) -> bool:
        return reduction(word[0] in text for word in ("a", ""))

    schema = {"text": STR}
    rows = [("a",), ("z",)]
    lowered = backend.lower(trace(reduced, schema).value)
    assert lowered.exact is not None
    expected = [True, False] if reduction is any else [False, True]
    assert backend.evaluate([lowered.exact], schema, rows)[0] == expected
    assert check_function(reduced, schema, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_list_reduction_keeps_eager_faults(backend: Backend[Any]) -> None:
    def reduced(text: str) -> bool:
        return any([word[0] in text for word in ("a", "")])

    assert check_function(reduced, {"text": STR}, backend, [("a",), ("z",)]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("reduction", [any, all])
def test_generator_expression_size_is_linear(
    backend: Backend[Any], reduction: Callable[[Iterable[bool]], bool]
) -> None:
    words: tuple[str, ...] = ()

    def reduced(text: str) -> bool:
        return reduction(word != text for word in words if word in text)

    sizes = []
    expressions = []
    for count in (32, 64):
        words = tuple(str(index) for index in range(count))
        value = trace(reduced, {"text": STR}).value
        pending = [value]
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            assert size <= 12 * count
            if isinstance(node, Call):
                pending.extend(node.arguments)
            elif isinstance(node, If):
                pending.extend((node.condition, node.then, node.orelse))
            elif isinstance(node, Guarded):
                pending.extend((node.value, node.exact))
        sizes.append(size)
        expressions.append(len(str(backend.lower(value).expr)))
    assert sizes[1] <= 2 * sizes[0] + 2
    assert expressions[1] <= 2 * expressions[0] + 100


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("reduction", [any, all])
def test_filtered_generator_keeps_fault_order(
    backend: Backend[Any], reduction: Callable[[Iterable[bool]], bool]
) -> None:
    def reduced(text: str) -> bool:
        return reduction(word[0] in text for word in ("a", "", "z") if word != text)

    assert check_function(reduced, {"text": STR}, backend, [("a",), ("z",), ("",)]) == []
