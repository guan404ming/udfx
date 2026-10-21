from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import ClassVar, cast

import pyarrow as pa
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.types import StructField, StructType

from udfx.backends import arrow
from udfx.backends.base import RESERVED, Backend, Row, output_names
from udfx.backends.registration import RuleSet
from udfx.backends.spark.dtypes import dtype
from udfx.backends.spark.rules import RULES
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Kind, Type

EXACT = f"{RESERVED}exact"


def fallback_patch(program: Program) -> Callable[[pa.Array], pa.Array]:
    result = arrow.dtype(program.result)

    def patch(batch: pa.Array) -> pa.Array:
        packed = cast(pa.StructArray, batch)
        columns = [packed.field(name) for name in program.schema]
        return arrow.fallback(SparkBackend(), program, packed.field(EXACT), columns, result)

    return patch


class SparkBackend(Backend[Column]):
    name: ClassVar[str] = "spark"
    rules: ClassVar[RuleSet[Column]] = RULES

    def __init__(self, session: SparkSession | None = None) -> None:
        self._session = session

    def session(self) -> SparkSession:
        if self._session is not None:
            if self._session.sparkContext._jsc is None:  # pyright: ignore[reportPrivateUsage]
                raise RuntimeError("the SparkSession given to SparkBackend was stopped")
            return self._session
        try:
            return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        except Exception as error:
            raise RuntimeError("udfx needs a SparkSession to build Spark columns") from error

    def fold_name(self, name: str) -> str:
        if self.session().conf.get("spark.sql.caseSensitive") == "true":
            return name
        return name.lower()

    def column(self, node: Col) -> Column:
        self.session()
        return functions.col("`" + node.name.replace("`", "``") + "`")

    def literal(self, node: Lit) -> Column:
        self.session()
        value = node.value
        if node.type.kind is Kind.LIST and value is not None:
            value = list(cast("Iterable[object]", value))
        if node.type.kind is Kind.DATETIME and isinstance(value, datetime):
            value = value.isoformat(" ")
        return functions.lit(value).cast(dtype(node.type))

    def select(self, condition: Column, then: Column, orelse: Column) -> Column:
        return functions.when(condition, then).otherwise(orelse)

    def list_value(self, items: Sequence[Column], type: Type) -> Column:
        return functions.array(*items).cast(dtype(type))

    def list_transform(self, source: Column, transform: Callable[[Column], Column]) -> Column:
        return functions.transform(source, transform)

    def list_filter(self, source: Column, predicate: Callable[[Column], Column]) -> Column:
        return functions.filter(source, predicate)

    def fallback(self, native: Column, exact: Column, program: Program) -> Column:
        result = dtype(program.result)
        flag = functions.coalesce(exact, functions.lit(False))
        packed = functions.struct(
            flag.alias(EXACT),
            *(self.column(Col(name, type)) for name, type in program.schema.items()),
        )
        slow = functions.arrow_udf(fallback_patch(program), result)(packed)  # pyright: ignore[reportCallIssue, reportArgumentType]
        return functions.when(flag, native).otherwise(slow)

    def evaluate(
        self, exprs: Sequence[Column], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        self.check_names(schema)
        struct = StructType([StructField(name, dtype(type)) for name, type in schema.items()])
        frame = self.session().createDataFrame(list(rows), struct)
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        collected = self.execute(lambda: frame.select(*aliased).collect())
        return [[row[index] for row in collected] for index in range(len(exprs))]
