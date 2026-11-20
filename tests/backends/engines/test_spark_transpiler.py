from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest
from conftest import spark_session
from pyspark.errors import PySparkException
from pyspark.sql import SparkSession
from pyspark.sql.types import BooleanType, DataType, LongType, StringType
from pyspark.sql.udf import UserDefinedFunction

from udfx.backends.engines.spark import transpiler

pytestmark = pytest.mark.xdist_group("spark")


def score(value: int) -> int:
    return value * 3 + 7 if value > 0 else -value


def normalize(value: str) -> str:
    return value.strip().lower()


def shout(value: str) -> str:
    return value.upper()


def positive(value: float) -> bool:
    return value * 2 > 1


def initials(value: str) -> str:
    return ",".join(word[:1] for word in value.split())


@pytest.mark.parametrize(
    ("function", "category", "result", "column", "values"),
    [
        (score, "integer", LongType(), "int", [1, -5, 0, 1_000_000_000]),
        (score, "integer", LongType(), "bigint", [3074457345618258600, -3074457345618258603]),
        (normalize, "string", StringType(), "string", ["  A@Ex.COM ", "\u2003\u0391\u03a3\t", "x"]),
        (shout, "string", StringType(), "string", ["straße", "ŉ ﬃ", "abc"]),
        (positive, "float", BooleanType(), "double", [0.5, 0.75, -1.0]),
    ],
)
def test_transpiled_option_matches_python(
    function: Callable[[Any], object],
    category: str,
    result: DataType,
    column: str,
    values: list[object],
) -> None:
    session = spark_session(ansi=True)
    expression = transpiler.lower(function, ["value"], {0: category}, result, session)
    assert expression is not None
    frame = session.createDataFrame([(value,) for value in values], f"_udf_param_0 {column}")
    assert [row[0] for row in frame.select(expression).collect()] == [
        function(value) for value in values
    ]


def test_transpiled_option_raises_where_python_raises() -> None:
    session = spark_session(ansi=True)
    expression = transpiler.lower(score, ["value"], {0: "integer"}, LongType(), session)
    assert expression is not None
    frame = session.createDataFrame([(None,)], "_udf_param_0 bigint")
    with pytest.raises(PySparkException, match="score"):
        frame.select(expression).collect()


@pytest.mark.parametrize("value", [3074457345618258601, -(2**63)])
def test_transpiled_option_overflows_with_ansi_arithmetic(value: int) -> None:
    session = spark_session(ansi=True)
    expression = transpiler.lower(score, ["value"], {0: "integer"}, LongType(), session)
    assert expression is not None
    assert "try_" not in str(expression)
    frame = session.createDataFrame([(value,)], "_udf_param_0 bigint")
    with pytest.raises(PySparkException, match="ARITHMETIC_OVERFLOW"):
        frame.select(expression).collect()


@pytest.mark.parametrize(
    ("function", "parameters", "categories", "result"),
    [
        (score, ["value"], {0: "binary"}, LongType()),
        (score, ["value"], {0: "integer"}, StringType()),
        (score, ["other"], {0: "integer"}, LongType()),
        (initials, ["value"], {0: "string"}, StringType()),
    ],
)
def test_transpiler_declines_unsupported_options(
    function: Callable[[Any], object],
    parameters: list[str],
    categories: dict[int, str],
    result: DataType,
) -> None:
    session = spark_session(ansi=True)
    assert transpiler.lower(function, parameters, categories, result, session) is None


def test_install_runs_python_udfs_natively() -> None:
    pytest.importorskip(transpiler.MODULE)
    session = spark_session(ansi=True)
    with patch.object(SparkSession, "_instantiatedSession", session):
        transpiler.install(session)
        udf = UserDefinedFunction(score, LongType())
    assert session.conf.get(transpiler.TRANSPILERS) == "udfx,catalyst"
    frame = session.range(-3, 3).select(udf("id").alias("score"))
    plan = StringIO()
    with redirect_stdout(plan):
        frame.explain()
    assert "EvalPython" not in plan.getvalue()
    assert [row.score for row in frame.collect()] == [score(value) for value in range(-3, 3)]
