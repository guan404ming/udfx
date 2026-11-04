from typing import Any

import pytest
from conftest import backend_params
from pyspark.sql import functions

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.frontend.core.splitting import single_sorted_token
from udfx.ir import Call, Col, lit
from udfx.ops import OpKey, apply
from udfx.testing import check_function
from udfx.types import STR, Kind, list_of

SPARK = [parameter for parameter in backend_params() if str(parameter.id).startswith("spark")]


@pytest.mark.parametrize("backend", SPARK)
def test_single_tokens_skip_split_and_sort(
    backend: Backend[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def ordered(values: list[str]) -> list[str]:
        return [" ".join(sorted(value.split(" "))) for value in values]

    original = backend.rules.get

    def rule(key: OpKey) -> Any:
        if key == OpKey(".split", (Kind.STR, Kind.STR)):
            return lambda *arguments, **facts: functions.raise_error("unexpected split").cast(
                "array<string>"
            )
        return original(key)

    monkeypatch.setattr(backend.rules, "get", rule)
    schema = {"values": list_of(STR)}
    rows = [([],), (["", "abc", "é", "漢", "😀", "a\x00b"],)]
    program = trace(ordered, schema)
    result = backend.build(program)
    assert backend.evaluate([result], schema, rows)[0] == program.run(rows)


@pytest.mark.parametrize("backend", SPARK)
@pytest.mark.parametrize("separator", [" ", "::", "aa"])
def test_sorted_tokens_keep_delimiters(backend: Backend[Any], separator: str) -> None:
    def ordered(values: list[str]) -> list[str]:
        return [separator.join(sorted(value.split(separator))) for value in values]

    rows = [
        (None,),
        ([],),
        (["", separator, separator * 3, "aaaaa"],),
        ([separator.join(["β", "", "a", "😀", "a\x00b"])],),
    ]
    assert check_function(ordered, {"values": list_of(STR).or_none()}, backend, rows) == []


@pytest.mark.parametrize("shape", ["empty", "different", "dynamic", "limited", "computed"])
def test_single_token_rule_keeps_other_splits(shape: str) -> None:
    value = Col("value", STR)
    separator = lit("") if shape == "empty" else lit(" ")
    if shape == "dynamic":
        separator = Col("separator", STR)
    if shape == "computed":
        value = apply(".lower", value)
    arguments = (value, separator, lit(2)) if shape == "limited" else (value, separator)
    split = apply(".split", *arguments)
    joined = lit(":") if shape == "different" else separator
    node = apply(".join", joined, apply("sorted", split))
    assert isinstance(node, Call)
    assert single_sorted_token(node) is None
