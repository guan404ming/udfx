import pytest
from conftest import spark_session

from udfx.backends.engines.spark import SparkBackend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import FLOAT

VALUES = (
    None,
    0.0,
    -0.0,
    1e-3,
    1.5e-4,
    1e-5,
    1e7,
    123456789.125,
    9999999999999998.0,
    1e16,
    1e22,
    -1.2345e-300,
    5e-324,
    1e-323,
    1.7976931348623157e308,
    float("nan"),
    float("inf"),
    float("-inf"),
)


def text(value: float | None) -> str:
    return str(value)


@pytest.mark.xdist_group("spark")
@pytest.mark.parametrize("ansi", [True, False])
def test_spark_float_text_needs_no_python(ansi: bool) -> None:
    backend = SparkBackend(spark_session(ansi=ansi))
    schema = {"value": FLOAT.or_none()}
    assert backend.lower_program(trace(text, schema)).exact is None
    assert check_function(text, schema, backend, [(value,) for value in VALUES]) == []
