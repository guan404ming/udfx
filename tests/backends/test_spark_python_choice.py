import pytest
from conftest import spark_session

from udfx.backends.diagnostics import capture_lowering
from udfx.backends.spark import SparkBackend
from udfx.testing import check_function
from udfx.types import STR

pytestmark = pytest.mark.xdist_group("spark")


def floats(text: str) -> list[float]:
    return [float(word) for word in text.split(" ")]


def test_guarded_lambdas_run_python() -> None:
    backend = SparkBackend(spark_session(ansi=True).newSession())
    with capture_lowering() as details:
        assert check_function(floats, {"text": STR}, backend, [("1 2.5",), ("-0 1e3",)]) == []
    assert details[-1].exact is False
