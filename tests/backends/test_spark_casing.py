import pytest
from pyspark.sql import Column, functions

from udfx.backends.spark import SparkBackend, strings
from udfx.ir import Col
from udfx.semantics.text import CaseDrift
from udfx.types import STR


@pytest.mark.parametrize("method", ["lower", "upper", "casefold"])
@pytest.mark.parametrize("alone", [None, r"[\x{1C89}]"])
def test_ascii_case_skips_unicode_guard(
    monkeypatch: pytest.MonkeyPatch, method: str, alone: str | None
) -> None:
    backend = SparkBackend()
    value = backend.column(Col("value", STR))
    monkeypatch.setattr(strings, "case_drift", lambda method: CaseDrift(alone, r"[\x{345}]"))
    monkeypatch.setattr(
        strings,
        "case_guard",
        lambda *args: functions.raise_error("Unicode guard evaluated").cast("boolean"),
    )
    rows = [("",), ("AbC",), ("aZ\n_09",)]
    expression = strings.recase(value, method)
    assert isinstance(expression, Column)
    assert backend.evaluate([expression], {"value": STR}, rows)[0] == [
        getattr(text, method)() for (text,) in rows
    ]


def test_ascii_case_keeps_ascii_drift_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SparkBackend()
    value = backend.column(Col("value", STR))
    monkeypatch.setattr(strings, "case_drift", lambda method: CaseDrift(r"[\x{41}]", None))
    monkeypatch.setattr(strings, "case_guard", lambda *args: functions.lit(False))
    monkeypatch.setattr(strings, "case_expression", lambda *args: functions.lit("incorrect"))
    expression = strings.recase(value, "lower")
    assert isinstance(expression, Column)
    assert backend.evaluate([expression], {"value": STR}, [("A",)])[0] == ["a"]
