from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from conftest import backend_params

from udfx import compile
from udfx.backends.base import Backend, FallbackError
from udfx.backends.polars import PolarsBackend
from udfx.ir import FALSE, Col, Lit
from udfx.program import Program
from udfx.testing import check_function, check_translation
from udfx.types import BOOL, FLOAT, INT, INT64_MAX, INT64_MIN, STR, list_of

from .functions import inverse, row_inverse, row_year, year

FLOATS = [(1.0,), (-1.0,), (0.0,), (-0.0,), (float("inf"),), (float("-inf"),), (float("nan"),)]


class CustomError(Exception):
    pass


class PairError(Exception):
    def __init__(self, left: str, right: str) -> None:
        super().__init__(left, right)


def test_polars_execute_restores_the_original_exception() -> None:
    original = PairError("left", "right")

    def fail(value: int) -> int:
        raise original

    backend = PolarsBackend()
    program = Program(fail, {"value": INT}, Col("value", INT))
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    frame = pl.DataFrame({"value": [1]})
    with pytest.raises(PairError) as raised:
        backend.execute(lambda: frame.select(expr))
    assert raised.value is original


def custom(x: int) -> int:
    raise CustomError(f"custom failure {x}")


def decode(x: int) -> str:
    return bytes([x]).decode()


def below_int64(x: int) -> bool:
    return x < 2**63


def above_int64(x: int) -> bool:
    return x > -(2**63) - 1


def constant_branch(value: int) -> int:
    if value == 0:
        return 1 // 0
    return 2


def literal_after_division(x: int) -> int:
    _ = 1 // x
    return 7


def split(s: str | None) -> list[str] | None:
    return s.split(",") if s else None


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_with_literal_branch_on_no_rows(backend: Backend[Any]) -> None:
    translation = compile(literal_after_division, {"x": INT}, backend)
    assert backend.evaluate([translation.expr], {"x": INT}, []) == [[]]
    assert check_function(literal_after_division, {"x": INT}, backend, [(1,), (0,)]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_evaluate_keeps_row_count_without_columns(backend: Backend[Any]) -> None:
    program = Program(lambda: 1, {}, Lit(1, INT))
    assert backend.evaluate([backend.build(program)], {}, [(), (), ()]) == [[1, 1, 1]]


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_raises_the_python_exception(backend: Backend[Any]) -> None:
    assert check_function(inverse, {"x": INT}, backend, [(2,), (0,), (4,)]) == []


def test_rebuilt_fallback_error_keeps_the_python_exception() -> None:
    error = FallbackError(ZeroDivisionError("division by zero"))
    rebuilt = PolarsBackend().unwrap(FallbackError(str(error)))
    assert isinstance(rebuilt, ZeroDivisionError)
    assert str(rebuilt) == "division by zero"


@pytest.mark.xdist_group("polars")
@pytest.mark.parametrize("plan", ["eager", "eager-pair", "lazy", "streaming"])
def test_polars_raises_without_execute(plan: str) -> None:
    translation = compile(inverse, {"x": INT}, PolarsBackend())
    assert translation.expr is not None
    frame = pl.DataFrame({"x": [2, 0, 4]})
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        if plan == "eager":
            frame.select(translation.expr)
        elif plan == "eager-pair":
            frame.select(translation.expr.alias("first"), translation.expr.alias("second"))
        elif plan == "lazy":
            frame.lazy().select(translation.expr).collect()
        else:
            frame.lazy().select(translation.expr).collect(engine="streaming")


@pytest.mark.parametrize("backend", backend_params())
def test_evaluate_forgets_earlier_python_exception(backend: Backend[Any]) -> None:
    with pytest.raises((ZeroDivisionError, RuntimeError)):
        backend.run_fallback_rows(Program(inverse, {"x": INT}, Col("x", INT)), [(0,)])
    with pytest.raises(Exception) as error:
        backend.evaluate([backend.column(Col("missing", INT))], {"x": INT}, [(1,)])
    assert not isinstance(error.value, ZeroDivisionError)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("function", "message"),
    [(custom, "custom failure 255"), (decode, "invalid start byte")],
    ids=["custom", "decode"],
)
def test_fallback_surfaces_exceptions_it_cannot_rebuild(
    backend: Backend[Any], function: Callable[[int], object], message: str
) -> None:
    program = Program(function, {"x": INT}, Col("x", INT))
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    with pytest.raises(Exception, match=message):
        backend.evaluate([expr], {"x": INT}, [(255,)])


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_patches_inexact_rows(backend: Backend[Any]) -> None:
    rows = [("2024-01",), (" 123",), ("\uff11\uff12\uff13\uff14",), ("9223372036854775807",)]
    assert check_function(year, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_patches_list_rows(backend: Backend[Any]) -> None:
    schema = {"text": STR, "exact": BOOL}
    result = list_of(STR)
    program = Program(lambda text, exact: text.split(), schema, Lit(["native"], result))
    expr = backend.fallback(
        backend.literal(Lit(["native"], result)), backend.column(Col("exact", BOOL)), program
    )
    rows = [("a b", False), ("c", True), ("", False)]
    assert backend.evaluate([expr], schema, rows) == [[["a", "b"], ["native"], []]]


@pytest.mark.parametrize("backend", backend_params())
def test_row_fallback_patches_inexact_rows(backend: Backend[Any]) -> None:
    schema = {"offset": INT, "text": STR, "unused": INT}
    rows = [(1, "2024", 99), (2, "\uff11\uff12\uff13\uff14", 98)]
    translation = compile(row_year, schema, backend)
    assert translation.program is not None, translation.unsupported
    exact = backend.lower(translation.program.value).exact
    assert exact is not None
    assert backend.evaluate([exact], schema, rows)[0][1] is False
    assert backend.evaluate([translation.expr], schema, rows) == [[2025, 1236]]
    assert check_translation(translation, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(("value", "kind"), [("0", ZeroDivisionError), ("invalid", ValueError)])
def test_row_fallback_restores_exceptions(
    backend: Backend[Any], value: str, kind: type[Exception]
) -> None:
    schema = {"value": STR}
    translation = compile(row_inverse, schema, backend)
    assert translation.ok, translation.unsupported
    with pytest.raises(kind):
        backend.evaluate([translation.expr], schema, [("2",), (value,)])


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_patches_rows_without_columns(backend: Backend[Any]) -> None:
    assert check_function(lambda: int("\uff11\uff12"), {}, backend, [(), ()]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_fallback_broadcasts_constant_inputs(backend: Backend[Any]) -> None:
    assert check_function(constant_branch, {"value": INT}, backend, [(1,), (1,), (0,)]) == []
    program = Program(lambda value: value + 1, {"value": INT}, Col("value", INT))
    expr = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    assert backend.evaluate([expr], {"value": INT}, [(1,), (2,)]) == [[2, 3]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("function", [below_int64, above_int64], ids=lambda f: f.__name__)
def test_fallback_runs_literals_outside_int64(
    backend: Backend[Any], function: Callable[[int], bool]
) -> None:
    rows = [(0,), (INT64_MAX,), (INT64_MIN,)]
    assert check_function(function, {"x": INT}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_nullable_list_literal(backend: Backend[Any]) -> None:
    rows = [("a,b",), ("",), (None,)]
    assert check_function(split, {"s": STR.or_none()}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda x: x if x > 0.0 else -0.0,
        lambda x: x == float("inf"),
        lambda x: x != float("nan"),  # noqa: PLW0177
    ],
    ids=["negative_zero", "infinity", "nan"],
)
def test_float_literals(backend: Backend[Any], function: Callable[[float], object]) -> None:
    assert check_function(function, {"x": FLOAT}, backend, FLOATS) == []
