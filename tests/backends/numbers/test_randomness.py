import random
from collections.abc import Callable
from typing import Any

import cloudpickle
import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.ops import Op, all_ops
from udfx.types import FLOAT, INT, Type


def default_noise() -> float:
    return random.gauss()


def mean_noise(mu: float) -> float:
    return random.gauss(mu)


def scaled_noise(mu: float, sigma: float) -> float:
    return random.gauss(mu, sigma)


def noise(value: int | None) -> float | None:
    return float(value) + random.gauss(0, 2) if value else None


def discarded(value: int) -> int:
    random.gauss()
    return value


@pytest.mark.parametrize("function", [noise, discarded])
def test_trace_preserves_random_state(function: Callable[..., object]) -> None:
    state = random.getstate()
    try:
        random.seed(43)
        before = random.getstate()
        program = trace(function, {"value": INT.or_none()})
        assert random.getstate() == before
        assert not program.deterministic
        rows = [(1,), (0,), (None,), (2,)]
        expected = [function(*row) for row in rows]
        after = random.getstate()
        random.setstate(before)
        restored = cloudpickle.loads(cloudpickle.dumps(program))
        assert restored.run(rows) == expected
        assert random.getstate() == after
    finally:
        random.setstate(state)


@pytest.mark.parametrize("backend", backend_params())
def test_random_calls_run_for_each_row(backend: Backend[Any]) -> None:
    translation = compile(noise, {"value": INT.or_none()}, backend)
    assert translation.ok, translation.unsupported
    assert translation.program is not None
    assert not translation.program.deterministic
    rows = [(1,)] * 64 + [(0,), (None,)]
    values = backend.evaluate([translation.expr], {"value": INT.or_none()}, rows)[0]
    assert values[-2:] == [None, None]
    assert len(set(values[:-2])) == 64
    assert all(isinstance(value, float) for value in values[:-2])


@pytest.mark.parametrize("backend", backend_params())
def test_constant_random_arguments_do_not_fold(backend: Backend[Any]) -> None:
    translation = compile(lambda: random.gauss(mu=4.0, sigma=0.0), {}, backend)
    assert translation.ok, translation.unsupported
    assert translation.program is not None
    assert translation.program.result == FLOAT
    assert not translation.program.deterministic
    assert backend.evaluate([translation.expr], {}, [(), (), ()]) == [[4.0, 4.0, 4.0]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("op", [op for op in all_ops() if not op.deterministic], ids=str)
def test_nondeterministic_signatures(
    backend: Backend[Any], op: Op, record_property: Callable[[str, object], None]
) -> None:
    record_property("backend", backend.name)
    record_property("signature", str(op))
    record_property("dimensions", "random-state,per-row,fallback")
    function: Callable[..., float] = (default_noise, mean_noise, scaled_noise)[len(op.parameters)]
    schema: dict[str, Type] = dict(zip(("mu", "sigma"), op.parameters, strict=False))
    row = tuple({"bool": True, "int": 1, "float": 1.0}[kind.kind] for kind in op.parameters)
    translation = compile(function, schema, backend)
    assert translation.ok, translation.unsupported
    assert translation.program is not None and not translation.program.deterministic
    values = backend.evaluate([translation.expr], schema, [row] * 16)[0]
    assert len(set(values)) == 16
    state = random.getstate()
    try:
        expected = [function(*row) for _ in range(16)]
        after = random.getstate()
        random.setstate(state)
        assert translation.program.run([row] * 16) == expected
        assert random.getstate() == after
    finally:
        random.setstate(state)
