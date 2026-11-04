import pytest

from udfx.backends.engines.polars import PolarsBackend
from udfx.frontend import trace
from udfx.ir import Col, Lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, STR, list_of


@pytest.mark.parametrize(
    "values",
    [["a", "b", "a"], ["😀", "é", "\x00"]],
)
def test_literal_index_uses_first_position(values: list[str | None]) -> None:
    def index(item: str | None) -> int:
        try:
            return values.index(item)
        except ValueError:
            return -1

    backend = PolarsBackend()
    schema = {"item": STR.or_none()}
    program = trace(index, schema)
    rows = [(item,) for item in (*values, "missing", "", None)]
    assert check_function(index, schema, backend, rows) == []
    assert "replace_strict" in str(backend.build(program))
    assert "list.eval" not in str(backend.build(program))


@pytest.mark.parametrize("values", [(), None])
def test_literal_index_handles_faulting_container(values: tuple[str, ...] | None) -> None:
    backend = PolarsBackend()
    item = Col("item", STR.or_none())
    value = apply(".index", Lit(values, list_of(STR)), item)
    result = backend.lower(value)
    columns = [result.expr, backend.column(item)]
    assert backend.evaluate(columns, {"item": STR.or_none()}, [("a",), (None,)])[0] == [0, 0]


def test_literal_index_preserves_guarded_faults() -> None:
    values: list[str | None] = ["a", "b", "a"]

    def index(item: str | None, enabled: bool) -> int:
        if not enabled:
            return -2
        try:
            return values.index(item)
        except ValueError:
            return -1

    backend = PolarsBackend()
    schema = {"item": STR.or_none(), "enabled": BOOL}
    rows = [("a", True), ("missing", False), (None, False), ("missing", True), (None, True)]
    assert check_function(index, schema, backend, rows) == []
    assert "replace_strict" in str(backend.build(trace(index, schema)))


def test_dynamic_index_keeps_list_evaluation() -> None:
    backend = PolarsBackend()
    node = apply(".index", Col("values", list_of(STR)), Col("item", STR))
    assert "list.eval" in str(backend.lower(node).expr)
