import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.lowering_context import bindings_enabled
from udfx.backends.program_lowering import lower_program
from udfx.backends.spark import backend as spark
from udfx.backends.spark import json as spark_json
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


def numeric_reuse(text: str) -> int:
    if text.capitalize() != text:
        return -1
    count = len(text.split("-"))
    return count if count > 1 else count + 1


def text_reuse(text: str) -> str:
    if text.capitalize() != text:
        return "other"
    count = len(text.split("-"))
    return str(count) if count > 1 else str(count + 1)


def nested_reuse(text: str) -> int:
    if text.capitalize() != text:
        return -1
    first = len(text.split("-"))
    second = len(text.split(":"))
    if first > 1:
        return first + second + second if second > 1 else first
    return first


def comprehension_reuse(text: str) -> int:
    if text.capitalize() != text:
        return -1
    count = len(text.split("-"))
    return sum([count + len(word) for word in text.split(":")]) if count > 1 else count


def unrelated_comprehension(text: str) -> int:
    if text.capitalize() != text:
        return -1
    count = len(text.split("-"))
    return count + len([word for word in text.split(":")]) if count > 1 else count


def native_reuse(text: str) -> int:
    if text.capitalize() != text:
        return -1
    count = len(text.split("-"))
    return count + len(json.dumps(text.split(":"))) if count > 1 else count


@pytest.mark.parametrize(
    "backend", [value for value in backend_params() if value.id in {"spark", "spark-ansi-off"}]
)
@pytest.mark.parametrize(
    ("function", "unbound"),
    [
        (numeric_reuse, False),
        (text_reuse, True),
        (nested_reuse, True),
        (comprehension_reuse, True),
        (unrelated_comprehension, False),
        (native_reuse, True),
        (native_reuse, False),
    ],
)
def test_guarded_bindings_keep_only_numeric_reuse(
    backend: spark.SparkBackend,
    function: Callable[..., object],
    unbound: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if function is native_reuse and unbound:
        monkeypatch.setattr(spark_json, "call", lambda *arguments: None)
    original = lower_program
    passes: list[bool] = []

    def lower(*arguments: Any, **keywords: Any) -> Any:
        passes.append(bindings_enabled(False))
        return original(*arguments, **keywords)

    monkeypatch.setattr(spark, "lower_program", lower)
    schema = {"text": STR}
    backend.build(trace(function, schema))
    assert passes == ([True, False] if unbound else [True])
    rows = [(value,) for value in ("", "A", "a", "A-b", "A-b:c", "ß-a", "ɤ:a-b", "中-a:b")]
    assert check_function(function, schema, backend, rows) == []
