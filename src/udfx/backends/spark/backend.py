from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import ClassVar, cast

import pyarrow as pa
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.types import StructField, StructType

from udfx.backends import arrow
from udfx.backends.base import RESERVED, Backend, Row, output_names
from udfx.backends.faults import NativeFaults, faults_for
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.spark.dtypes import dtype
from udfx.backends.spark.rules import RULES
from udfx.errors import Unsupported
from udfx.ir import Col, Comprehension, Element, Lit, Node
from udfx.program import Program
from udfx.types import Kind, Type

EXACT = f"{RESERVED}exact"
NATIVE = f"{RESERVED}native"


def fallback_patch(program: Program) -> Callable[[pa.Array], pa.Array]:
    def patch(batch: pa.Array) -> pa.Array:
        packed = cast(pa.StructArray, batch)
        columns = [packed.field(name) for name in program.schema]
        return arrow.fallback(
            SparkBackend(), program, packed.field(EXACT), packed.field(NATIVE), columns
        )

    return patch


class SparkBackend(Backend[Column]):
    name: ClassVar[str] = "spark"
    rules: ClassVar[RuleSet[Column]] = RULES

    def __init__(self, session: SparkSession | None = None) -> None:
        self._session = session

    def lower(
        self, node: Node, elements: Mapping[Element, Column] | None = None
    ) -> Lowered[Column]:
        lowered = super().lower(node, elements)
        if isinstance(node, Comprehension) and elements and lowered.exact is not None:
            raise Unsupported("Spark cannot pass guarded nested lambdas to a Python UDF")
        return lowered

    def session(self) -> SparkSession:
        if self._session is not None:
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

    @property
    def native_faults(self) -> NativeFaults:
        return faults_for(self.session().sparkContext)

    def native_raise(self, program: Program) -> Column:
        packed = functions.struct(
            *(
                self.column(Col(name, type)).alias(str(index))
                for index, (name, type) in enumerate(program.schema.items())
            )
        )
        encoded = functions.hex(
            functions.encode(
                functions.to_json(packed, {"timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"}),
                "UTF-8",
            )
        )
        message = functions.concat(functions.lit(self.native_faults.marker(program)), encoded)
        return functions.raise_error(message).cast(dtype(program.result))

    def fallback(self, native: Column, exact: Column, program: Program) -> Column:
        flag = functions.coalesce(exact, functions.lit(False))
        packed = functions.struct(
            flag.alias(EXACT),
            functions.when(~flag, native).alias(NATIVE),
            *(
                functions.when(~flag, self.column(Col(name, type))).alias(name)
                for name, type in program.schema.items()
            ),
        )
        patched = functions.arrow_udf(fallback_patch(program), dtype(program.result))(packed)  # pyright: ignore[reportCallIssue, reportArgumentType]
        return functions.when(flag, native.cast(dtype(program.result))).otherwise(patched)

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
