from collections.abc import Callable

import pytest

from udfx import compile
from udfx.backends.polars import PolarsBackend
from udfx.types import STR

OFFSET = 1
ROWS = [("12",), ("\uff11\uff12",)]


def shifted(text: str) -> int:
    return int(text) + OFFSET


def nested(text: str) -> int:
    def helper(value: str) -> int:
        return int(value) + OFFSET

    return helper(text)


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
