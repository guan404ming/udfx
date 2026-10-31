from dataclasses import replace

import pytest
from pyspark.sql import functions

from udfx.backends.lowering_context import lowering_context
from udfx.backends.spark import SparkBackend, lists
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.types import STR, list_of


def test_frequency_indices_only_cover_distinct_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lists, "call", lambda *arguments: None)
    original = lists.FREQUENCY_OFFSETS.indices
    primitives = replace(
        lists.FREQUENCY_OFFSETS,
        indices=lambda count: functions.when(
            count > 3, functions.raise_error("Full source indices allocated")
        ).otherwise(original(count)),
    )
    monkeypatch.setattr(lists, "FREQUENCY_OFFSETS", primitives)
    backend = SparkBackend()
    source = Col("values", list_of(STR))
    with lowering_context(binds=True):
        lowered = backend.lower(apply("frequency_order", source, lit(5)))
    rows = [(["b", "a", "b"] * 50 + ["c"],)]
    assert backend.evaluate([lowered.expr], {"values": list_of(STR)}, rows) == [[["b", "a", "c"]]]
