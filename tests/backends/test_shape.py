from collections.abc import Callable
from dataclasses import replace

import polars as pl
import pytest

from udfx import compile
from udfx.backends.polars import PolarsBackend
from udfx.backends.polars.rules import broadcast
from udfx.backends.polars.strings import join
from udfx.testing import check_translation
from udfx.types import STR, Type


@pytest.mark.xdist_group("polars")
@pytest.mark.parametrize("plan", ["eager", "lazy", "with_columns"])
@pytest.mark.parametrize("height", [0, 3])
@pytest.mark.parametrize(
    ("function", "schema", "value"),
    [
        (lambda text: "k", {"text": STR.or_none()}, "k"),
        (lambda text: None, {"text": STR.or_none()}, None),
        (lambda text: ["k"], {"text": STR.or_none()}, ["k"]),
        (lambda: "k", {}, "k"),
    ],
    ids=["string", "null", "list", "no_columns"],
)
def test_polars_constant_keeps_frame_height(
    function: Callable[..., object],
    schema: dict[str, Type],
    value: object,
    height: int,
    plan: str,
) -> None:
    translation = compile(function, schema)
    assert translation.expr is not None, translation.unsupported
    expr = translation.expr.alias("out")
    frame = pl.DataFrame({"text": [None, "a", "b"]}, schema={"text": pl.String}).head(height)
    if not schema:
        frame = frame.drop("text")
    if plan == "eager":
        output = frame.select(expr)
    elif plan == "lazy":
        output = frame.lazy().select(expr).collect()
    else:
        output = frame.with_columns(expr)
    assert output.height == frame.height
    assert output.get_column("out").to_list() == [value] * height


@pytest.mark.parametrize("height", [0, 3])
def test_check_translation_rejects_wrong_row_count(height: int) -> None:
    translation = compile(lambda text: "k", {"text": STR.or_none()})
    translation = replace(translation, expr=pl.lit("k"))
    rows = [(None,), ("a",), ("b",)][:height]
    with pytest.raises(AssertionError, match=f"polars returned 1 rows for {height} input rows"):
        check_translation(translation, PolarsBackend(), rows)


@pytest.mark.parametrize("height", [0, 1, 3])
def test_polars_broadcast_skips_length_preserving_expressions(height: int) -> None:
    frame = pl.DataFrame({"value": [1, 2, 3]}).head(height)
    column = pl.col("value")
    assert broadcast(column) is column
    assert frame.select(broadcast(column)).height == height
    expression = column + 1
    assert broadcast(expression) is expression
    assert frame.select(broadcast(expression)).to_series().to_list() == [2, 3, 4][:height]
    scalar = column.first().fill_null(0)
    assert scalar.meta.root_names() == ["value"]
    assert frame.select(broadcast(scalar)).to_series().to_list() == [1] * height


@pytest.mark.parametrize("height", [0, 1, 3])
def test_polars_join_broadcasts_scalar_with_column_roots(height: int) -> None:
    frame = pl.DataFrame({"words": [["a", "b"], ["c"], []], "separator": [",", "-", " "]}).head(
        height
    )
    assert (
        frame.select(join(pl.col("separator"), pl.col("words"))).to_series().to_list()
        == ["a,b", "c", ""][:height]
    )
    assert (
        frame.select(join(pl.col("separator"), pl.col("words").first())).to_series().to_list()
        == ["a,b", "a-b", "a b"][:height]
    )
    evaluated = pl.col("words").list.eval(pl.element().str.to_uppercase())
    assert (
        frame.select(join(pl.col("separator"), evaluated)).to_series().to_list()
        == ["A,B", "C", ""][:height]
    )


def test_polars_list_transform_broadcasts_constant_body() -> None:
    backend = PolarsBackend()
    frame = pl.DataFrame({"values": [[1, None, 3], [], None]})
    assert pl.element().meta.root_names() == []
    expr = backend.list_transform(pl.col("values"), lambda element: pl.lit(7))
    assert frame.select(expr).to_series().to_list() == [[7, 7, 7], [], None]
