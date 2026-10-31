import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

import pytest
from conftest import spark_session
from pyspark.errors import PySparkException
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.udf import UDFRegistration

from bench.fingerprint import fingerprint
from udfx.backends.spark import SparkBackend, lists, runtime
from udfx.backends.spark import json as spark_json
from udfx.frontend import trace
from udfx.ir import Col
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import INT, INT64_MAX, INT64_MIN, STR, list_of

pytestmark = pytest.mark.xdist_group("spark")


def packed(text: str) -> str:
    return json.dumps(text.split())


def frequency(backend: SparkBackend) -> Column:
    return backend.lower(
        apply("frequency_order", Col("values", list_of(STR)), Col("count", INT))
    ).expr


def unexpected(*arguments: object, **keywords: object) -> Any:
    raise AssertionError("Unexpected engine expression path")


@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_json_runtime_replaces_word_encoding(
    ensure_ascii: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def encode(text: str) -> str:
        return json.dumps(text.split(), ensure_ascii=ensure_ascii)

    monkeypatch.setattr(spark_json, "encode_spaced_words", unexpected)
    backend = SparkBackend(spark_session(ansi=True).newSession())
    rows = [(None,), ("",), ('\x00 quote" slash\\ café\u2003😀\x7f',)]
    assert check_function(encode, {"text": STR.or_none()}, backend, rows) == []


def test_frequency_runtime_preserves_dynamic_int64_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lists, "ordered_frequencies_value", unexpected)
    monkeypatch.setattr(lists, "offset_frequencies", unexpected)
    backend = SparkBackend(spark_session(ansi=True).newSession())
    values = ["😀", "é", "😀", "é", "last"]
    rows = [(values, count) for count in (INT64_MIN, 0, 1, INT64_MAX)]
    assert backend.evaluate([frequency(backend)], {"values": list_of(STR), "count": INT}, rows) == [
        [[], [], ["😀"], ["😀", "é", "last"]]
    ]


def test_json_runtime_runs_inside_a_list_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    def encode(values: list[str]) -> list[str]:
        return [json.dumps(value.split()) for value in values]

    monkeypatch.setattr(spark_json, "encode_spaced_words", unexpected)
    backend = SparkBackend(spark_session(ansi=True).newSession())
    assert (
        check_function(
            encode, {"values": list_of(STR)}, backend, [([],), (["a b", "é\u2003😀", ""],)]
        )
        == []
    )


def test_runtime_accepts_null_fault_inputs() -> None:
    backend = SparkBackend(spark_session(ansi=True).newSession())
    with runtime.scope(backend.session):
        ordered = runtime.call(
            "frequency_order",
            functions.lit(None).cast("array<string>"),
            functions.lit(1).cast("long"),
        )
        encoded = runtime.call(
            "json_words",
            functions.lit(None).cast("string"),
            functions.lit(True),
            functions.lit(" "),
        )
    assert ordered is not None and encoded is not None
    assert backend.evaluate([ordered, encoded], {}, [()]) == [[None], [None]]


def test_runtime_registers_once_per_supplied_session() -> None:
    first = spark_session(ansi=True).newSession()
    second = first.newSession()
    original = UDFRegistration.registerJavaFunction
    calls: list[tuple[SparkSession, str]] = []

    def register(registration: UDFRegistration, name: str, class_name: str, result: Any) -> None:
        calls.append((registration.sparkSession, class_name))
        original(registration, name, class_name, result)

    with patch.object(UDFRegistration, "registerJavaFunction", register):
        for session in (first, first, second):
            backend = SparkBackend(session)
            expr = backend.build(trace(packed, {"text": STR}))
            assert backend.evaluate([expr], {"text": STR}, [("a b",)]) == [['["a", "b"]']]
    assert [session for session, _ in calls] == [first, second]
    assert all(name.endswith(".JsonWords") for _, name in calls)


def test_runtime_concurrent_registration_uses_one_function() -> None:
    session = spark_session(ansi=True).newSession()
    original = UDFRegistration.registerJavaFunction
    calls: list[str] = []

    def register(registration: UDFRegistration, name: str, class_name: str, result: Any) -> None:
        calls.append(name)
        original(registration, name, class_name, result)

    def build(index: int) -> Column:
        return frequency(SparkBackend(session))

    with (
        patch.object(UDFRegistration, "registerJavaFunction", register),
        ThreadPoolExecutor(4) as pool,
    ):
        expressions = list(pool.map(build, range(4)))
    assert len(calls) == 1
    assert len({str(expr) for expr in expressions}) == 1
    assert SparkBackend(session).evaluate(
        expressions[:1], {"values": list_of(STR), "count": INT}, [(["b", "a", "b"], 2)]
    ) == [[["b", "a"]]]


@pytest.mark.parametrize("message", ["denied", ""])
def test_runtime_registration_failure_keeps_engine_expressions(message: str) -> None:
    backend = SparkBackend(spark_session(ansi=True).newSession())
    program = trace(packed, {"text": STR})
    with (
        patch.object(
            UDFRegistration, "registerJavaFunction", side_effect=PySparkException(message)
        ) as register,
        pytest.warns(RuntimeWarning, match="runtime is unavailable") as warnings,
    ):
        expressions = [backend.build(program) for _ in range(2)]
    assert register.call_count == len(warnings) == 1
    assert all("udfx runtime" not in str(expr) for expr in expressions)
    assert (
        backend.evaluate(expressions, {"text": STR}, [("a b",), ("é 😀",)])
        == [['["a", "b"]', '["\\u00e9", "\\ud83d\\ude00"]']] * 2
    )


def test_runtime_registration_is_lazy() -> None:
    session = spark_session(ansi=True).newSession()
    with patch.object(UDFRegistration, "registerJavaFunction", side_effect=unexpected) as register:
        backend = SparkBackend(session)
        expr = backend.build(trace(lambda value: value + 1.0, {"value": INT}))
    assert register.call_count == 0
    assert backend.evaluate([expr], {"value": INT}, [(2,)]) == [[3.0]]


def test_runtime_fingerprint_is_stable_across_sessions() -> None:
    first = SparkBackend(spark_session(ansi=True).newSession())
    second = SparkBackend(first.session().newSession())
    program = trace(packed, {"text": STR})
    assert fingerprint(first, program) == fingerprint(second, program)
