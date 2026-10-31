import json

import pytest

from bench.fingerprint import fingerprint, normalize_lambdas
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.types import STR


@pytest.mark.parametrize(
    "text, expected",
    [
        ("transform(c0, x_17 -> x_17)", "transform(c0, element0 -> element0)"),
        (
            "aggregate(c0, 0, (x_4, y_5) -> x_4 + y_5)",
            "aggregate(c0, 0, (element0, element1) -> element0 + element1)",
        ),
        (
            "transform(c0, x_1 -> transform(c1, x_2 -> x_1 + x_2))",
            "transform(c0, element0 -> transform(c1, element1 -> element0 + element1))",
        ),
        (
            "transform(c0, x_1 -> concat(x_1, 'x_1 -> x_1', `x_1`))",
            "transform(c0, element0 -> concat(element0, 'x_1 -> x_1', `x_1`))",
        ),
        ("concat(x_1, 'x_1 -> x_1')", "concat(x_1, 'x_1 -> x_1')"),
        (r"concat('a\'x_1 -> x_1', x_1)", r"concat('a\'x_1 -> x_1', x_1)"),
        ("concat('a''x_1 -> x_1', x_1)", "concat('a''x_1 -> x_1', x_1)"),
    ],
)
def test_spark_fingerprint_keeps_literals(text: str, expected: str) -> None:
    assert normalize_lambdas(text) == expected


def test_spark_casefold_fingerprint_is_stable() -> None:
    backend = SparkBackend()
    first = trace(lambda value: value.casefold(), {"value": STR})
    second = trace(lambda text: text.casefold(), {"text": STR})
    assert fingerprint(backend, first) == fingerprint(backend, second)


def test_spark_prepared_fingerprint_is_stable() -> None:
    backend = SparkBackend()
    programs = [
        trace(
            lambda value, key: json.loads(value).get(key),
            {"value": STR, "key": STR},
            result=STR.or_none(),
        ),
        trace(
            lambda text, field: json.loads(text).get(field),
            {"text": STR, "field": STR},
            result=STR.or_none(),
        ),
        trace(
            lambda value, key: json.loads(value).get("value"),
            {"value": STR, "key": STR},
            result=STR.or_none(),
        ),
    ]
    assert backend.lower_program(programs[0]).prepare is not None
    first, second, third = [fingerprint(backend, program) for program in programs]
    assert all(value == first for value in (second, fingerprint(backend, programs[0])))
    assert first != third
