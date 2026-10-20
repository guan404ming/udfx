from collections.abc import Callable, Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type

POLARS = PolarsBackend()


def extractfunder(project: str | None) -> str | None:
    if project:
        try:
            if "::" in project:
                return project.split("::")[0]
            else:
                return None
        except:  # noqa: E722
            return None
    else:
        return None


def extractyear(arg: str | None) -> int | None:
    if arg:
        try:
            return int(arg[: arg.find("-")])
        except:  # noqa: E722
            return -1
    else:
        return None


def converttoeuro(x: str | None, y: str | None) -> float | None:
    rates = {"EUR": 1.0, "USD": 1.09, "GBP": 0.85}
    if x is not None and y is not None:
        try:
            return float(x) / rates[y]
        except:  # noqa: E722
            return 0.0
    else:
        return None


def removeshort(name: str) -> bool:
    def keep(word: str) -> bool:
        return len(word) > 2

    return keep(name)


TARGETS: list[tuple[Callable[..., object], Mapping[str, Type], Type]] = [
    (extractfunder, {"project": STR.or_none()}, STR.or_none()),
    (extractyear, {"arg": STR.or_none()}, INT.or_none()),
    (converttoeuro, {"x": STR.or_none(), "y": STR.or_none()}, FLOAT.or_none()),
    (removeshort, {"name": STR}, BOOL),
]


@pytest.mark.parametrize(("function", "schema", "result"), TARGETS)
def test_target_traces_without_faults(
    function: Callable[..., object], schema: Mapping[str, Type], result: Type
) -> None:
    program = trace(function, schema)
    assert program.result == result
    assert program.fault == FALSE


def test_removeshort_inlines_helper() -> None:
    program = trace(removeshort, {"name": STR})
    assert program.value == apply("__gt__", apply("len", Col("name", STR)), Lit(2, INT))


@pytest.mark.parametrize(("function", "schema", "result"), TARGETS)
@given(data=st.data())
def test_target_matches_python(
    function: Callable[..., object],
    schema: Mapping[str, Type],
    result: Type,
    data: st.DataObject,
) -> None:
    rows = data.draw(batches(schema))
    assert check_function(function, schema, POLARS, rows) == []
