from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import ClassVar, cast

import pyarrow as pa
from pyspark.errors import PythonException
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.pandas.types import to_arrow_type
from pyspark.sql.types import StructField, StructType

from udfx.backends.base import Backend, Row, RuleSet
from udfx.backends.spark.dtypes import dtype
from udfx.backends.spark.rules import RULES
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Kind, Type

EXACT = "udfx exact"
OUT = "udfx out"


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


def raised(error: PythonException) -> type[Exception] | None:
    name = str(error).rstrip().rpartition("\n")[2].partition(":")[0]
    kind = getattr(builtins, name, None)
    return kind if isinstance(kind, type) and issubclass(kind, Exception) else None


class SparkBackend(Backend[Column]):
    name: ClassVar[str] = "spark"
    rules: ClassVar[RuleSet[Column]] = RULES

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
        arrow_type = to_arrow_type(result)

        def run(batch: pa.Array) -> pa.Array:  # pyright: ignore[reportUnknownParameterType]
            flags = batch.field(EXACT).to_pylist()
            columns = [batch.field(name).to_pylist() for name in names]
            pending = [index for index, flag in enumerate(flags) if not flag]
            rows = ([column[index] for column in columns] for index in pending)
            values: list[object] = [None] * len(flags)
            for index, value in zip(pending, program.run(rows), strict=True):
                values[index] = value
            return pa.array(values, type=arrow_type)

        flag = functions.coalesce(exact, functions.lit(False))
        packed = functions.struct(
            flag.alias(EXACT),
            *(self.column(Col(name, type)) for name, type in program.schema.items()),
        )
        slow = functions.arrow_udf(run, result)(packed)  # pyright: ignore[reportCallIssue, reportArgumentType]
        return functions.when(flag, native).otherwise(slow)

    def evaluate(
        self, exprs: Sequence[Column], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        struct = StructType([StructField(name, dtype(type)) for name, type in schema.items()])
        frame = session().createDataFrame(list(rows), struct)
        names = [f"{OUT} {index}" for index in range(len(exprs))]
        query = frame.select(*(expr.alias(name) for expr, name in zip(exprs, names, strict=True)))
        try:
            collected = query.collect()
        except PythonException as error:
            kind = raised(error)
            if kind is None:
                raise
            raise kind(str(error)) from error
        return [[row[index] for row in collected] for index in range(len(exprs))]
