import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.core.logic import conjunction, negate
from udfx.ir import FALSE, Col, If, Lit
from udfx.ops import apply
from udfx.testing import batches, check_function
from udfx.types import INT, STR

POLARS = PolarsBackend()

S = Col("s", STR.or_none())
MISSING = apply("is_none", S)
SCHEMA = {"s": STR.or_none()}


def catch_attribute(s: str) -> str:
    try:
        return s.lower()
    except AttributeError:
        return "none"


def catch_tuple(s: str) -> str:
    try:
        return s.lower()
    except (ValueError, AttributeError):
        return "none"


def catch_named(s: str) -> str:
    try:
        return s.lower()
    except Exception as error:
        del error
        return "none"


def catch_all(s: str) -> str:
    try:
        return s.lower()
    except:  # noqa: E722
        return "none"


def catch_other(s: str) -> str:
    try:
        return s.lower()
    except ValueError:
        return "none"


def catch_second(s: str) -> str:
    try:
        return s.lower()
    except ValueError:
        return "value"
    except AttributeError:
        return "attribute"


def catch_nested(s: str) -> str:
    try:
        try:
            return s.lower()
        except ValueError:
            return "inner"
    except AttributeError:
        return "none"


def catch_to_none(s: str) -> str | None:
    try:
        return s.upper()
    except AttributeError:
        return None


def catch_to_int(s: str) -> str | int:
    try:
        return s.upper()
    except AttributeError:
        return 0


def outside(s: str) -> str:
    if not s:
        return s.lower()
    try:
        return s.lower()
    except AttributeError:
        return "none"


def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1


def with_finally(s: str) -> str:
    try:
        return s.lower()
    finally:
        pass


CAUGHT = (catch_attribute, catch_tuple, catch_named, catch_all, catch_nested)


def test_handler_catches_fault() -> None:
    expected = If(MISSING, Lit("none", STR), apply(".lower", S), STR)
    for function in CAUGHT:
        program = trace(function, SCHEMA)
        assert program.value == expected, function.__name__
        assert program.fault == FALSE, function.__name__


def test_unmatched_handler_reraises() -> None:
    program = trace(catch_other, SCHEMA)
    assert program.value == apply(".lower", S)
    assert program.fault == MISSING


def test_second_handler_matches() -> None:
    program = trace(catch_second, SCHEMA)
    assert program.value == If(MISSING, Lit("attribute", STR), apply(".lower", S), STR)


def test_handler_return_type_joins_unify() -> None:
    program = trace(catch_to_none, SCHEMA)
    assert program.result == STR.or_none()
    with pytest.raises(Unsupported, match="differ"):
        trace(catch_to_int, SCHEMA)


def test_fault_outside_try_raises() -> None:
    program = trace(outside, SCHEMA)
    assert program.fault == conjunction(negate(apply("bool", S)), MISSING)


def test_finally_reraises() -> None:
    program = trace(with_finally, SCHEMA)
    assert program.value == apply(".lower", S)
    assert program.fault == MISSING


def test_op_fault_is_caught() -> None:
    s = Col("s", STR)
    program = trace(parse, {"s": STR})
    valid = apply("int_parses", s)
    assert program.value == If(valid, apply("int", s), Lit(-1, INT), INT)
    assert program.fault == FALSE


@given(batches(SCHEMA))
def test_handlers_match_python(rows: list[tuple[object, ...]]) -> None:
    functions = (*CAUGHT, catch_other, catch_second, catch_to_none, outside, with_finally)
    for function in functions:
        assert check_function(function, SCHEMA, POLARS, rows) == [], function.__name__
