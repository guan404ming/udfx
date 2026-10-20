from hypothesis import given

from udfx import compile
from udfx.backends.polars import PolarsBackend
from udfx.testing import batches, check_function
from udfx.types import INT, STR

POLARS = PolarsBackend()


def test_unsupported_is_reported() -> None:
    translation = compile(lambda s: s.casefold(), {"s": STR})
    assert not translation.ok
    assert translation.unsupported is not None


@given(batches({"s": STR.or_none()}))
def test_nullable_lower_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(lambda s: s.lower(), {"s": STR.or_none()}, POLARS, rows) == []


@given(batches({"a": INT, "b": INT}))
def test_overflow_falls_back(rows: list[tuple[object, ...]]) -> None:
    assert check_function(lambda a, b: a + b > 0, {"a": INT, "b": INT}, POLARS, rows) == []
