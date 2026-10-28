import json
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.polars import PolarsBackend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import OpKey, apply
from udfx.semantics.text import PREDICATES
from udfx.testing import check_function
from udfx.testing.text_traps import STRING_TRAPS
from udfx.types import INT, INT64_MAX, INT64_MIN, JSON, STR, Kind, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("default", [None, "missing"])
def test_json_get_skips_presence_check_only_for_none(
    backend: Backend[Any], default: str | None
) -> None:
    def lookup(text: str, key: str) -> str | None:
        return json.loads(text).get(key, default)

    schema = {"text": STR, "key": STR}
    program = trace(lookup, schema, result=STR.or_none())
    expression = backend.build(program)
    if backend.name == "spark":
        value = apply(
            "json.get", Col("text", JSON), Col("key", STR), lit(json.dumps(default), JSON)
        )
        assert ("json_object_keys" in str(backend.lower(value).expr)) == (default is not None)
    rows = [
        ("{}", "name"),
        ('{"name": null}', "name"),
        ('{"name": "value"}', "name"),
        ('{"": null}', ""),
        ('{"quote\'": "value"}', "quote'"),
    ]
    assert backend.evaluate([expression], schema, rows) == [[lookup(*row) for row in rows]]


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("name", ["__eq__", "__ne__"])
def test_integer_equality_uses_nonnullable_operands_once(backend: Backend[Any], name: str) -> None:
    value = apply(name, Col("first_operand", INT), Col("second_operand", INT))
    expression = str(backend.lower(value).expr)
    assert expression.count("first_operand") == 1
    assert expression.count("second_operand") == 1


@pytest.mark.parametrize("backend", [DuckDBBackend(), PolarsBackend()], ids=["duckdb", "polars"])
def test_reverse_uses_guarded_native_expression(backend: Backend[Any]) -> None:
    def reverse(text: str) -> str:
        return text[::-1]

    schema = {"text": STR}
    lowered = backend.lower(trace(reverse, schema).value)
    expression = str(lowered.expr)
    assert expression.count('"text"') == 1
    if backend.name == "duckdb":
        assert 'THEN (reverse("udfx item")) ELSE COALESCE(array_to_string' in expression
        assert lowered.exact is None
    else:
        assert expression == 'col("text").str.reverse()'
        assert lowered.exact is not None
    assert check_function(reverse, schema, backend, [(value,) for value in STRING_TRAPS]) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [lambda text: text.replace("-", ""), lambda text: text.split("_")],
    ids=["replace", "split"],
)
def test_literal_text_branches_disappear(
    backend: Backend[Any], function: Callable[[str], object]
) -> None:
    schema = {"text": STR}
    lowered = backend.lower(trace(function, schema).value)
    expression = str(lowered.expr).lower()
    if isinstance(backend, SparkBackend):
        frame = backend.session().createDataFrame([("one_two",)], ["text"])
        output = StringIO()
        with redirect_stdout(output):
            frame.select(lowered.expr.alias("result")).explain()
        expression = output.getvalue().lower()
    assert "regexp_extract_all" not in expression
    assert "str.extract_all" not in expression
    assert "is_null" not in expression
    assert "is null" not in expression
    assert "regexp_replace" not in expression
    if backend.name == "polars":
        assert ".when(" not in expression
    assert check_function(function, schema, backend, [(value,) for value in STRING_TRAPS]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_string_truth_uses_byte_length(backend: Backend[Any]) -> None:
    lowered = backend.lower(apply("bool", Col("text", STR.or_none())))
    expression = str(lowered.expr).lower()
    assert any(name in expression for name in ("len_bytes", "strlen", "octet_length"))


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("index", [INT64_MIN, -2, -1, 0, 1, INT64_MAX])
def test_literal_string_item_uses_slice(
    backend: Backend[Any], index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def selected(text: str) -> str:
        return text[index]

    def forbidden(*arguments: object) -> Any:
        pytest.fail("Lowered a literal string index as a dynamic index")

    original = backend.rules.get
    indexed = OpKey("__getitem__", (Kind.STR, Kind.INT))
    monkeypatch.setattr(
        backend.rules, "get", lambda key: forbidden if key == indexed else original(key)
    )
    samples = [None, "", "a", "abc", "😀a\u0301", "\x00尾"]
    assert check_function(selected, {"text": STR.or_none()}, backend, [(x,) for x in samples]) == []


@pytest.mark.parametrize("backend", backend_params())
def test_literal_replacement_allows_faulted_null_arguments(backend: Backend[Any]) -> None:
    schema = {"text": STR}
    for old, new in [(None, "x"), ("a", None), ("", None)]:
        lowered = backend.lower(apply(".replace", Col("text", STR), lit(old, STR), lit(new, STR)))
        backend.evaluate([lowered.expr], schema, [("abc",), ("",)])


@pytest.mark.parametrize("backend", [value for value in backend_params() if value.id == "spark"])
def test_spark_predicates_use_ascii_branch(backend: Backend[Any]) -> None:
    schema = {"text": STR}
    rows = [(value,) for value in STRING_TRAPS]
    expressions = []
    for method in PREDICATES:
        lowered = backend.lower(apply(f".{method}", Col("text", STR)))
        assert lowered.exact is None
        expression = str(lowered.expr)
        if method == "isascii":
            assert "rlike" not in expression
        else:
            fast, unicode = expression.split(" ELSE ", 1)
            assert "octet_length(text)" in fast
            assert "char_length(text)" in fast
            assert fast.count("rlike(") == (2 if method == "istitle" else 1)
            assert len(fast) < 200
            assert "rlike(" in unicode
        expressions.append(lowered.expr)
    actual = backend.evaluate(expressions, schema, rows)
    assert actual == [[getattr(value, method)() for value in STRING_TRAPS] for method in PREDICATES]


@pytest.mark.parametrize("backend", backend_params())
def test_integer_parse_uses_one_validation_regex(backend: Backend[Any]) -> None:
    program = trace(lambda text: int(text[:2]), {"text": STR})
    parsed = backend.lower(program.value)
    validation = backend.lower(apply("int_parses", Col("text", STR)))
    expression = str(parsed.expr).lower()
    assert parsed.exact is None
    assert parsed.overflow is None
    assert "from_json" not in expression
    assert "to_json" not in expression
    assert "cast(" in expression
    if backend.name == "duckdb":
        assert "regexp_replace(" not in expression
    for operation in ("regexp_matches(", "regexp_like(", "rlike(", "str.contains("):
        assert operation not in expression
    assert (
        sum(
            str(validation.expr).lower().count(operation)
            for operation in ("regexp_matches(", "regexp_like(", "rlike(", "str.contains(")
        )
        == 1
    )
    literal = backend.lower(apply("int", lit("12\x00")))
    backend.evaluate([literal.expr], {"text": STR}, [("",)])


@pytest.mark.parametrize(
    "backend", [DuckDBBackend(), DataFusionBackend()], ids=["duckdb", "datafusion"]
)
def test_list_index_uses_source_once(backend: Backend[Any]) -> None:
    words = apply(".split", Col("text", STR))
    expression = str(backend.lower(apply("__getitem__", words, Col("index", INT))).expr)
    assert expression.count("text") == 1
    assert "cardinality(" not in expression
    assert "len(" not in expression


def test_polars_column_join_skips_concatenation() -> None:
    items = Col("items", list_of(STR.or_none()))
    expression = PolarsBackend().lower(apply(".join", lit(","), items)).expr
    assert ".list.concat(" not in str(expression)


def test_datafusion_sum_binds_input_per_expression() -> None:
    lowered = DataFusionBackend().lower(apply("sum", Col("items", list_of(INT))))
    assert str(lowered.expr).count("items") == 1
    assert str(lowered.exact).count("items") == 1
