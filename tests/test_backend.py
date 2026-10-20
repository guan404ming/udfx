from typing import Any

import pytest

from udfx.backends import installed
from udfx.backends.base import Backend
from udfx.ir import Lit
from udfx.program import Program
from udfx.testing import check_function
from udfx.types import INT, STR

BACKENDS = installed()


def case_id(backend: Backend[Any]) -> str:
    return backend.name


def inverse(x: int) -> float:
    return 1 / x


def year(text: str) -> int:
    return int(text[:4]) + 1


@pytest.mark.parametrize("backend", BACKENDS, ids=case_id)
def test_evaluate_keeps_row_count_without_columns(backend: Backend[Any]) -> None:
    program = Program(lambda: 1, {}, Lit(1, INT))
    assert backend.evaluate([backend.build(program)], {}, [(), (), ()]) == [[1, 1, 1]]


@pytest.mark.parametrize("backend", BACKENDS, ids=case_id)
def test_fallback_raises_the_python_exception(backend: Backend[Any]) -> None:
    assert check_function(inverse, {"x": INT}, backend, [(2,), (0,), (4,)]) == []


@pytest.mark.parametrize("backend", BACKENDS, ids=case_id)
def test_fallback_patches_inexact_rows(backend: Backend[Any]) -> None:
    rows = [("2024-01",), (" 123",), ("\uff11\uff12\uff13\uff14",), ("9223372036854775807",)]
    assert check_function(year, {"text": STR}, backend, rows) == []
