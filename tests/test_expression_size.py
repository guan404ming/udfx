from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops.registry import apply
from udfx.types import INT, STR, Type, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "schema", "size"),
    [
        (lambda s: s // 60 // 60 // 24 // 7, {"s": INT}, 6679),
        (lambda t: t[1:][1:][1:][1:], {"t": STR}, 500),
        (lambda a, b: min(max(min(max(a, b), a), b), a), {"a": INT, "b": INT}, 116),
    ],
    ids=["division", "slice", "extrema"],
)
def test_expression_size(
    backend: Backend[Any], function: Callable[..., object], schema: dict[str, Type], size: int
) -> None:
    translation = compile(function, schema, backend)
    assert translation.ok
    assert len(str(translation.expr)) < 2 * size


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("start", "stop"),
    [(None, 4), (1, None), (1, 4), (None, -1), (-4, None), (1, -1), (-4, -1), (-4, 4)],
)
@pytest.mark.parametrize("step", [None, 1, -1, 2, -2])
def test_nested_literal_slices(
    backend: Backend[Any], start: int | None, stop: int | None, step: int | None
) -> None:
    node = Col("text", STR)
    sizes = []
    for _ in range(8):
        arguments = [node, lit(start, INT.or_none()), lit(stop, INT.or_none())]
        if step is not None:
            arguments.append(lit(step))
        node = apply("slice", *arguments)
        expr = str(backend.lower(node).expr)
        assert expr.count("text") == 1
        sizes.append(len(expr))
    assert sizes[-1] <= 8 * sizes[0]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("step", [None, 1])
def test_unit_step_is_plain_slice(backend: Backend[Any], step: int | None) -> None:
    arguments = (Col("text", STR), lit(None, INT.or_none()), lit(-1))
    plain = backend.lower(apply("slice", *arguments))
    stepped = backend.lower(apply("slice", *arguments, lit(step, INT.or_none())))
    assert str(stepped.expr) == str(plain.expr)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("length", [18, 19])
@pytest.mark.parametrize("step", [(), (1,), (-1,), (2,)])
def test_bounded_integer_guard(backend: Backend[Any], length: int, step: tuple[int, ...]) -> None:
    text = apply(
        "slice",
        Col("text", STR),
        lit(None, INT.or_none()),
        lit(length),
        *(lit(value) for value in step),
    )
    lowered = backend.lower(apply("int", text))
    assert (lowered.exact is None) == (length <= 18 and step in ((), (1,)))


@pytest.mark.parametrize("backend", backend_params())
def test_nested_unpacking_uses_source_once(backend: Backend[Any]) -> None:
    def unpack(items: list[str]) -> list[str]:
        _first, *middle, _last = items
        _first, *middle, _last = middle
        _first, *middle, _last = middle
        _first, *middle, _last = middle
        return middle

    program = trace(unpack, {"items": list_of(STR)})
    expr = str(backend.lower(program.value).expr)
    assert expr.count("items") == 1
    assert len(expr) < 1000
