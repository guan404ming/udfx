import json
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
from pyspark.sql import functions

from udfx.backends.registration_json import register_json_values
from udfx.backends.rules import RuleSet
from udfx.backends.spark import SparkBackend
from udfx.backends.spark import json as spark_json
from udfx.backends.spark.conversions import TEXT
from udfx.frontend import trace
from udfx.ir import Lit
from udfx.ops import OpKey
from udfx.semantics.json_encoding import encode_strings
from udfx.semantics.text_predicates import ascii_uppercase_free
from udfx.testing import check_function
from udfx.types import STR, Kind, list_of


def test_lowered_json_skips_escape_tokens() -> None:
    def packed(value: str) -> str:
        values = [" ".join(sorted(word.lower().split(" "))) for word in value.split("|")]
        return json.dumps(sorted(values))

    primitives = replace(
        spark_json.ENCODING,
        tokens=lambda *arguments: functions.raise_error("Unexpected escape tokens").cast(
            "array<string>"
        ),
    )
    rules = RuleSet()
    backend = SparkBackend()
    register_json_values(
        rules,
        backend.literal,
        TEXT,
        encode_strings=partial(encode_strings, primitives),
    )
    key = OpKey("json.dumps", (Kind.LIST, Kind.BOOL))
    rows = [
        ("",),
        ("ABC DEF|GHI",),
        ("ÉCOLE 漢字 😀|İSTANBUL ΣΣ",),
        ('quote" slash\\ zero\x00 delete\x7f|ABC',),
        (r"\uABCD \\uFFFF|ABC",),
    ]
    original = backend.rules.get
    with patch.object(
        backend.rules, "get", lambda found: rules.get(key) if found == key else original(found)
    ):
        assert check_function(packed, {"value": STR}, backend, rows) == []
        assert backend.lower_program(trace(packed, {"value": STR})).exact is None


@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("suffix", ["ABC", "É"])
def test_json_encoding_keeps_unproven_case(ensure_ascii: bool, suffix: str) -> None:
    def packed(value: str) -> str:
        return json.dumps([value.lower(), suffix], ensure_ascii=ensure_ascii)

    rows = [("漢字 😀",), (r"\uABCD",), ("İSTANBUL ΣΣ",)]
    assert check_function(packed, {"value": STR}, SparkBackend(), rows) == []


@pytest.mark.parametrize("value,expected", [([None, "é"], True), (["ABC"], False)])
def test_case_fact_accepts_list_literals(value: list[str | None], expected: bool) -> None:
    assert ascii_uppercase_free(Lit(value, list_of(STR.or_none()))) is expected
