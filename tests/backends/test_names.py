from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import SparkSession

from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.errors import Unsupported
from udfx.ir import FALSE, Col, Lit
from udfx.ops import apply
from udfx.program import Program
from udfx.types import INT

CASE_INSENSITIVE = {"duckdb", "spark"}


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    ("first", "second", "folding"),
    [("a", "A", {"duckdb", "spark"}), ("\u00e9", "\u00c9", {"spark"})],
    ids=["ascii", "accented"],
)
def test_column_names_that_differ_only_by_case(
    backend: Backend[Any], first: str, second: str, folding: set[str]
) -> None:
    schema = {first: INT, second: INT}
    value = apply("__sub__", Col(first, INT), Col(second, INT))
    program = Program(lambda left, right: left - right, schema, value)
    if backend.name in folding:
        with pytest.raises(Unsupported) as error:
            backend.build(program)
        assert str(error.value) == (
            f"{backend.name} column names are case-insensitive: {first!r} and {second!r} collide"
        )
    else:
        assert backend.evaluate([backend.build(program)], schema, [(5, 3)]) == [[2]]


@pytest.mark.xdist_group("spark")
def test_spark_case_sensitive_session_keeps_case() -> None:
    session = SparkSession.builder.getOrCreate().newSession()
    session.conf.set("spark.sql.caseSensitive", "true")
    assert SparkBackend(session).fold_name("A") == "A"


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["Amount", "first name", "select", "order", 'a"b', "a.b", "a`b"])
def test_column_names_that_need_quoting(backend: Backend[Any], name: str) -> None:
    program = Program(lambda value: value, {name: INT}, Col(name, INT))
    native = backend.build(program)
    slow = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
    assert backend.evaluate([native, slow], {name: INT}, [(5,)]) == [[5], [5]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["udfx row", "udfx exact", "udfx out 0", "UDFX ROW"])
def test_reserved_column_names(backend: Backend[Any], name: str) -> None:
    schema = {name: INT}
    program = Program(lambda value: value, schema, Col(name, INT))
    if name.islower() or backend.name in CASE_INSENSITIVE:
        with pytest.raises(Unsupported, match="column names starting with 'udfx ' are reserved"):
            backend.build(program)
        with pytest.raises(Unsupported, match="column names starting with 'udfx ' are reserved"):
            backend.evaluate([backend.literal(Lit(1, INT))], schema, [(5,)])
    else:
        native = backend.build(program)
        slow = backend.fallback(backend.literal(Lit(0, INT)), backend.literal(FALSE), program)
        assert backend.evaluate([native, slow], schema, [(5,), (7,)]) == [[5, 7], [5, 7]]
