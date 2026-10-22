import pytest

from udfx import compile
from udfx.backends.duckdb import DuckDBBackend
from udfx.ir import Col
from udfx.ops import apply
from udfx.types import STR


@pytest.mark.parametrize("operation", ["difference", "symmetric_difference"])
def test_duckdb_set_difference_binds_operands(operation: str) -> None:
    operands = [apply(".split", Col(name, STR)) for name in ("left", "right")]
    expression = str(DuckDBBackend().lower(apply(f"set_{operation}", *operands)).expr)
    assert expression.count("regexp_extract_all(") == 2
    assert "regexp_extract_all(" not in expression.split("list_filter(")[1]
    assert expression.count('"left"') == expression.count('"right"') == 1


def test_duckdb_set_size_uses_unique_count() -> None:
    characters = apply("list", Col("text", STR))
    expression = str(DuckDBBackend().lower(apply("set_size", characters)).expr)
    assert expression.count("list_unique(") == 1
    assert "list_distinct(" not in expression


def test_duckdb_runs_set_operations_in_python() -> None:
    def overlap(left: str, right: str) -> int:
        return len(set(left.split()) & set(right.split()))

    backend = DuckDBBackend()
    schema = {"left": STR, "right": STR}
    expression = compile(overlap, schema, backend).expr
    assert expression is not None
    assert "list_intersect(" not in str(expression)
    assert backend.evaluate([expression], schema, [("a b", "b c"), ("", "a")]) == [[1, 0]]


def test_duckdb_keeps_simple_set_sizes_native() -> None:
    expression = compile(lambda text: len(set(text)), {"text": STR}, DuckDBBackend()).expr
    assert "udfx fallback" not in str(expression)
