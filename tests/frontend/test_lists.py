import pytest
from hypothesis import given

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.invocation import invoke
from udfx.ir import FALSE, Col, Lit
from udfx.ops import Op, apply, register
from udfx.ops.registry import scoped
from udfx.testing import batches, check_function
from udfx.types import INT, STR, Type, list_of

POLARS = PolarsBackend()

STRICT_ITEMS = list_of(STR)

OPTIONAL_ITEMS = list_of(STR.or_none())


def first_item(items: list[str | None]) -> str | None:
    return items[0]


def joined(items: list[str]) -> str:
    return ",".join(items)


def test_nullable_items_do_not_bind_to_strict_items() -> None:
    with scoped():
        register(Op(".probe", (STRICT_ITEMS,), INT, len))
        _, faults = invoke(".probe", [Col("items", STRICT_ITEMS)], AttributeError)
        assert faults == []
        with pytest.raises(Unsupported, match=r"no op \.probe\(list\[str\?\]\)"):
            invoke(".probe", [Col("items", OPTIONAL_ITEMS)], AttributeError)


@pytest.mark.parametrize(
    ("items", "result"), [(STRICT_ITEMS, STR), (OPTIONAL_ITEMS, STR.or_none())], ids=str
)
def test_item_keeps_its_nullability(items: Type, result: Type) -> None:
    assert trace(first_item, {"items": items}).result == result


def test_join_of_nullable_items_faults_on_none() -> None:
    items = Col("items", OPTIONAL_ITEMS)
    program = trace(joined, {"items": OPTIONAL_ITEMS})
    assert program.fault == apply("__contains__", items, Lit(None, STR.or_none()))
    assert trace(joined, {"items": STRICT_ITEMS}).fault == FALSE


@given(batches({"items": OPTIONAL_ITEMS}))
def test_nullable_items_match_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(first_item, {"items": OPTIONAL_ITEMS}, POLARS, rows) == []
    assert check_function(joined, {"items": OPTIONAL_ITEMS}, POLARS, rows) == []
