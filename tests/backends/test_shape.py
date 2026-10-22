from collections.abc import Callable
from dataclasses import replace

import polars as pl
import pytest

from udfx import compile
from udfx.backends.polars import PolarsBackend
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
