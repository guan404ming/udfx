# ruff: noqa: PLW0108
from collections.abc import Callable
from types import FunctionType, ModuleType

import pytest
from pyspark import cloudpickle

from udfx import compile
from udfx.backends.polars import PolarsBackend
from udfx.types import STR

OFFSET = 1
OFFSETS = {"amount": 1}
ROWS = [("12",), ("\uff11\uff12",)]


def shifted(text: str) -> int:
    return int(text) + OFFSET


def nested(text: str) -> int:
    def helper(value: str) -> int:
        return int(value) + OFFSET

    return helper(text)


def global_helper(text: str) -> int:
    return shifted(text)


def mutable_helper(text: str) -> int:
    return int(text) + OFFSETS["amount"]


def default_helper(text: str, offsets: dict[str, int] = OFFSETS) -> int:
    return int(text) + offsets["amount"]


@pytest.mark.parametrize("function", [shifted, nested])
def test_globals_are_frozen(
    function: Callable[[str], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = PolarsBackend()
    translation = compile(function, {"text": STR}, backend)
    monkeypatch.setitem(globals(), "OFFSET", 2)
    assert translation.function is function
    assert translation.program is not None
    assert translation.expr is not None
    assert set(translation.program.function.__globals__) == {"__builtins__", "OFFSET"}
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]


def test_empty_closure_is_unsupported() -> None:
    offset = 1

    def shifted(text: str) -> int:
        return int(text) + offset

    assert shifted.__closure__ is not None
    del shifted.__closure__[0].cell_contents
    translation = compile(shifted, {"text": STR})
    assert str(translation.unsupported) == "unbound local offset"


def test_closure_cells_are_frozen() -> None:
    offset = 1

    def shifted(text: str) -> int:
        return int(text) + offset

    backend = PolarsBackend()
    translation = compile(shifted, {"text": STR}, backend)
    offset = 2
    assert translation.expr is not None
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]


def test_helper_globals_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = PolarsBackend()
    translation = compile(global_helper, {"text": STR}, backend)
    monkeypatch.setitem(globals(), "OFFSET", 2)
    assert translation.program is not None
    assert translation.expr is not None
    assert translation.program.run(ROWS) == [13, 13]
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]


@pytest.mark.parametrize("helper", [mutable_helper, default_helper])
def test_helper_mutable_globals_are_frozen(
    helper: Callable[[str], int], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = PolarsBackend()
    translation = compile(lambda text: helper(text), {"text": STR}, backend)
    monkeypatch.setitem(OFFSETS, "amount", 2)
    assert translation.program is not None
    assert translation.expr is not None
    assert translation.program.run(ROWS) == [13, 13]
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]


def test_module_helper_uses_its_own_frozen_globals() -> None:
    module = ModuleType("helpers")
    original = FunctionType(shifted.__code__, {"OFFSET": 4})
    vars(module)["helper"] = original
    backend = PolarsBackend()
    translation = compile(lambda text: module.helper(text), {"text": STR}, backend)
    assert module.helper is original
    module.helper.__globals__["OFFSET"] = 8
    assert translation.program is not None
    assert translation.expr is not None
    assert translation.program.run(ROWS) == [16, 16]
    assert cloudpickle.loads(cloudpickle.dumps(translation.program)).run(ROWS) == [16, 16]
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[16, 16]]


def test_helper_closure_is_frozen() -> None:
    offset = 1

    def helper(text: str) -> int:
        return int(text) + offset

    backend = PolarsBackend()
    translation = compile(lambda text: helper(text), {"text": STR}, backend)
    offset = 2
    assert translation.program is not None
    assert translation.expr is not None
    assert translation.program.run(ROWS) == [13, 13]
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]


def test_nested_helper_keeps_its_defining_globals() -> None:
    def invoke(helper: Callable[[str], int], text: str) -> int:
        return helper(text)

    foreign = FunctionType(invoke.__code__, {"OFFSET": 100})

    def function(text: str) -> int:
        def helper(value: str) -> int:
            return int(value) + OFFSET

        return foreign(helper, text)

    backend = PolarsBackend()
    translation = compile(function, {"text": STR}, backend)
    assert translation.program is not None
    assert translation.expr is not None
    assert translation.program.run(ROWS) == [13, 13]
    assert backend.evaluate([translation.expr], {"text": STR}, ROWS) == [[13, 13]]
