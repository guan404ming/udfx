from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import ClassVar, cast

import pyarrow as pa
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.types import StructField, StructType

from udfx.backends import arrow
from udfx.backends.base import Backend, Row, RuleSet
from udfx.backends.spark.dtypes import dtype
from udfx.backends.spark.rules import RULES
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Kind, Type

EXACT = "udfx exact"
OUT = "udfx out"
RAISED = "udfx raised "


@cache
def session() -> SparkSession:
    return (
        SparkSession.builder.master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


class SparkBackend(Backend[Column]):
    name: ClassVar[str] = "spark"
    rules: ClassVar[RuleSet[Column]] = RULES

    def patch(self, program: Program, rows: Iterable[Sequence[object]]) -> list[object]:
        try:
            return program.run(rows)
        except Exception as error:
            raise RuntimeError(f"{RAISED}{type(error).__name__}: {error}") from error

    def unwrap(self, error: BaseException) -> BaseException:
        _, found, detail = str(error).partition(RAISED)
        name, _, message = detail.partition(": ")
        kind = getattr(builtins, name.strip(), None)
        if not found or not isinstance(kind, type) or not issubclass(kind, Exception):
            return error
        return kind(message.partition("\n")[0])

    def column(self, node: Col) -> Column:
        session()
        return functions.col(node.name)

    def literal(self, node: Lit) -> Column:
        session()
        value = node.value
        if node.type.kind is Kind.LIST:
            value = list(cast("Iterable[object]", value))
        return functions.lit(value).cast(dtype(node.type))

    def select(self, condition: Column, then: Column, orelse: Column) -> Column:
        return functions.when(condition, then).otherwise(orelse)

    def both(self, left: Column, right: Column) -> Column:
        return left & right

    def negate(self, value: Column) -> Column:
        return ~value

    def fallback(self, native: Column, exact: Column, program: Program) -> Column:
        names = list(program.schema)
        result = dtype(program.result)
        arrow_type = arrow.dtype(program.result)

        def patch(batch: pa.Array) -> pa.Array:
            packed = cast(pa.StructArray, batch)
            columns = [packed.field(name) for name in names]
            pending = arrow.pending(packed.field(EXACT))
            values = self.fallback_values(program, len(packed), pending, arrow.gather(columns))
            return pa.array(values, type=arrow_type)

        flag = functions.coalesce(exact, functions.lit(False))
        packed = functions.struct(
            flag.alias(EXACT),
            *(self.column(Col(name, type)) for name, type in program.schema.items()),
        )
        slow = functions.arrow_udf(patch, result)(packed)  # pyright: ignore[reportCallIssue, reportArgumentType]
        return functions.when(flag, native).otherwise(slow)

    def evaluate(
        self, exprs: Sequence[Column], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        struct = StructType([StructField(name, dtype(type)) for name, type in schema.items()])
        frame = session().createDataFrame(list(rows), struct)
        names = [f"{OUT} {index}" for index in range(len(exprs))]
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        collected = self.execute(lambda: frame.select(*aliased).collect())
        return [[row[index] for row in collected] for index in range(len(exprs))]
