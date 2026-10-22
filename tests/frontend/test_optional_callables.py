# ruff: noqa: PLW0108

from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.resolve.callables import op_name
from udfx.ir import Call
from udfx.testing import check_function
from udfx.testing.strategies import FLOAT_TRAPS, INT_TRAPS
from udfx.types import BOOL, FLOAT, INT, STR, Type, fits

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")


@pytest.mark.parametrize("name", ["isna", "isnull", "notna", "notnull"])
@pytest.mark.parametrize(
    ("type", "values"),
    [(STR, ["", "nan"]), (INT, list(INT_TRAPS)), (FLOAT, list(FLOAT_TRAPS)), (BOOL, [False, True])],
)
def test_pandas_callable(name: str, type: Type, values: list[object]) -> None:
    function = getattr(pandas, name)

    def missing(value: object):
        return function(value)

    schema = {"value": type.or_none()}
    program = trace(missing, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == ("pandas.notna" if name.startswith("not") else "pandas.isna")
    assert (
        check_function(missing, schema, PolarsBackend(), [(value,) for value in [*values, None]])
        == []
    )


@pytest.mark.parametrize("name", ["isnan", "log", "sqrt", "exp", "abs", "round", "floor", "ceil"])
def test_numpy_callable(name: str) -> None:
    function = getattr(numpy, name)

    def calculate(value: object):
        return function(value)

    schema = {"value": FLOAT.or_none()}
    program = trace(calculate, schema)
    assert isinstance(program.value, Call)
    assert program.value.op.name == f"numpy.{name}"
    with numpy.errstate(all="ignore"):
        assert all(
            fits(value, program.result)
            for value in program.run([(value,) for value in FLOAT_TRAPS])
        )
        assert (
            check_function(
                calculate, schema, PolarsBackend(), [(value,) for value in (*FLOAT_TRAPS, None)]
            )
            == []
        )


@pytest.mark.parametrize(("type", "values"), [(INT, INT_TRAPS), (BOOL, (False, True))])
def test_numpy_isnan_integer_and_boolean(type: Type, values: tuple[object, ...]) -> None:
    def missing(value: object):
        return numpy.isnan(value)

    schema = {"value": type.or_none()}
    assert trace(missing, schema).result == BOOL
    assert (
        check_function(missing, schema, PolarsBackend(), [(value,) for value in (*values, None)])
        == []
    )


@pytest.mark.parametrize("name", ["log", "sqrt", "exp"])
def test_numpy_integer_conversion(name: str) -> None:
    function = getattr(numpy, name)

    def calculate(value: object):
        return function(value)

    program = trace(calculate, {"value": INT})
    assert program.result == FLOAT
    with numpy.errstate(all="ignore"):
        assert (
            check_function(
                calculate, {"value": INT}, PolarsBackend(), [(value,) for value in INT_TRAPS]
            )
            == []
        )


def test_numpy_round_digits() -> None:
    def rounded(value: float, digits: int):
        return numpy.round(value, digits)

    schema = {"value": FLOAT, "digits": INT}
    program = trace(rounded, schema)
    assert program.result == FLOAT
    rows = [(value, digits) for value in FLOAT_TRAPS for digits in (-2, 0, 2, 22, 308, 309)]
    with numpy.errstate(all="ignore"):
        assert check_function(rounded, schema, PolarsBackend(), rows) == []


@pytest.mark.parametrize("name", ["abs", "round", "floor", "ceil"])
def test_numpy_integer_result_is_unsupported(name: str) -> None:
    function = getattr(numpy, name)

    def calculate(value: object):
        return function(value)

    with pytest.raises(Unsupported, match="no op numpy"):
        trace(calculate, {"value": INT})


@pytest.mark.parametrize("name", ["log", "sqrt", "exp", "abs", "round", "floor", "ceil"])
def test_numpy_boolean_result_is_unsupported(name: str) -> None:
    function = getattr(numpy, name)

    def calculate(value: object):
        return function(value)

    with pytest.raises(Unsupported, match="no op numpy"):
        trace(calculate, {"value": BOOL})


@pytest.mark.parametrize(
    "function",
    [
        lambda value: numpy.isnan(value),
        lambda value: numpy.log(value),
        lambda value: numpy.sqrt(value),
        lambda value: numpy.exp(value),
        lambda value: numpy.abs(value),
        lambda value: numpy.round(value),
        lambda value: numpy.floor(value),
        lambda value: numpy.ceil(value),
    ],
)
def test_numpy_text_is_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match="no op numpy"):
        trace(function, {"value": STR})


def test_unrelated_callable_name_is_unsupported() -> None:
    def isna(value: object):
        return value is None

    with pytest.raises(Unsupported, match="unknown callable"):
        op_name(isna)


@pytest.mark.parametrize(
    "function",
    [
        lambda value: numpy.log(value, None),
        lambda value: numpy.round(value, 2, None),
        lambda value: numpy.round(value, decimals=2),
        lambda value: pandas.isna(value, True),
        lambda value: pandas.notna(value, True),
    ],
)
def test_optional_callable_extra_arguments(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"value": FLOAT})
