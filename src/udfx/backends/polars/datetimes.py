from __future__ import annotations

import polars as pl

from udfx.backends.polars.rules import RULES
from udfx.backends.registration import register_datetime_arithmetic, register_datetimes
from udfx.semantics.datetimes import (
    TIMESTAMP_LITERAL,
    DatetimeArithmeticPrimitives,
    DatetimePrimitives,
)


def part(value: pl.Expr, name: str) -> pl.Expr:
    if name == "epoch_day":
        return value.dt.date().cast(pl.Int64)
    if name == "doy":
        return value.dt.ordinal_day().cast(pl.Int64)
    return getattr(value.dt, name)().cast(pl.Int64)


register_datetimes(
    RULES,
    DatetimePrimitives(
        literal=pl.lit,
        part=part,
        pad=lambda value, width: value.cast(pl.String).str.pad_start(width, "0"),
        concatenate=pl.concat_str,
        extract=lambda text, pattern, group: text.str.extract(pattern, group),
        parse=lambda text: pl.when(text.str.contains(TIMESTAMP_LITERAL)).then(
            text.str.to_datetime("%Y-%m-%d %H:%M:%S%.6f", time_unit="us", strict=False)
        ),
        to_integer=lambda value: value.cast(pl.Int64, strict=False),
        select=lambda condition, then, orelse: pl.when(condition).then(then).otherwise(orelse),
        is_null=pl.Expr.is_null,
    ),
    pl.Expr.eq_missing,
)


register_datetime_arithmetic(
    RULES,
    DatetimeArithmeticPrimitives(
        literal=pl.lit,
        difference=lambda left, right: left.cast(pl.Int64) - right.cast(pl.Int64),
        to_microseconds=lambda value: value.cast(pl.Int64),
        from_microseconds=lambda value: value.cast(pl.Datetime("us")),
        select=lambda condition, then, orelse: pl.when(condition).then(then).otherwise(orelse),
    ),
)
