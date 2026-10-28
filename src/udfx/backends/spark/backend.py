from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import ClassVar, cast

import pyarrow as pa
from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.types import StructField, StructType

from udfx.backends import arrow
from udfx.backends.arithmetic import lower_integer_region
from udfx.backends.base import RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.faults import NativeFaults, faults_for
from udfx.backends.hoisting import free
from udfx.backends.indexing import literal_range
from udfx.backends.lowering_context import lowering_context
from udfx.backends.program_lowering import lower_program
from udfx.backends.registration import Lowered, RuleSet
from udfx.backends.results import ResultLowering
from udfx.backends.rules import Scope
from udfx.backends.spark.bindings import bind, binding_scope, reading
from udfx.backends.spark.dtypes import dtype
from udfx.backends.spark.numbers import WIDE_INTEGERS
from udfx.backends.spark.rules import RULES
from udfx.errors import Unsupported
from udfx.ir import Col, Comprehension, Lit, Node
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
    indexes: ClassVar[bool] = True
    graphs: ClassVar[bool] = True

    def __init__(self, session: SparkSession | None = None) -> None:
        self._session = session

    def lower_program(self, program: Program) -> Lowered[Column]:
        with binding_scope() as scope, lowering_context(binds=True) as context:
            lowered = lower_program(self, program, ResultLowering(self, program))
        if lowered.exact is not None and context.has_bindings and scope.requires_unbound:
            return lower_program(self, program, ResultLowering(self, program))
        return lowered

    def lower(self, node: Node, elements: Scope[Column] | None = None) -> Lowered[Column]:
        with reading(node, elements):
            lowered = super().lower(node, elements)
        if isinstance(node, Comprehension) and elements and self.exact(lowered) is not None:
            captured = free(node.value) | free(node.condition) | free(node.fault)
            if captured - {node.element}:
                raise Unsupported("Spark cannot pass guarded capturing lambdas to a Python UDF")
        return lowered

    def session(self) -> SparkSession:
        if self._session is not None:
            return self._session
        try:
            return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
        except Exception as error:
            raise RuntimeError("udfx needs a SparkSession to build Spark columns") from error

    def lower_node(self, node: Node, elements: Scope[Column]) -> Lowered[Column]:
        if isinstance(node, Comprehension) and (
            isinstance(node.source, Lit) or literal_range(node) is not None
        ):
            with lowering_context(binds=False):
                return super().lower_node(node, elements)
        if not elements or node not in elements:
            widened = lower_integer_region(self, node, elements, WIDE_INTEGERS)
            if widened is not None:
                return widened
        return super().lower_node(node, elements)

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

    def select_many(self, cases: Sequence[tuple[Column, Column]], orelse: Column) -> Column:
        selected = functions.when(*cases[0])
        for condition, value in cases[1:]:
            selected = selected.when(condition, value)
        return selected.otherwise(orelse)

    def list_value(self, items: Sequence[Column], type: Type) -> Column:
        return functions.array(*items).cast(dtype(type))

    def list_transform(self, source: Column, transform: Callable[[Column], Column]) -> Column:
        return functions.transform(source, transform)

    def list_filter(self, source: Column, predicate: Callable[[Column], Column]) -> Column:
        return functions.filter(source, predicate)

    def bind(self, value: Column, body: Callable[[Column], Column]) -> Column:
        return bind(value, body, tracked=True)

    def bind_sequence(self, value: Column, steps: Sequence[Callable[[Column], Column]]) -> Column:
        for step in steps:
            value = functions.aggregate(
                functions.array(functions.lit(0)), value, lambda total, item: total, step
            )
        return value

    def record(self, values: Sequence[Column]) -> Column:
        return functions.struct(*(value.alias(str(index)) for index, value in enumerate(values)))

    def field(self, value: Column, index: int) -> Column:
        return value[str(index)]

    def list_indexed(
        self,
        source: Column,
        value: Callable[[Column, Column], Column],
        condition: Callable[[Column, Column], Column] | None,
    ) -> Column:
        if condition is None:
            return functions.transform(
                source, lambda element, index: value(element, index.cast("long"))
            )
        pairs = functions.transform(
            source,
            lambda element, index: functions.struct(
                element.alias("element"), index.cast("long").alias("index")
            ),
        )
        if condition is not None:
            pairs = functions.filter(pairs, lambda pair: condition(pair["element"], pair["index"]))
        return functions.transform(pairs, lambda pair: value(pair["element"], pair["index"]))

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
        result = dtype(program.result)
        flag = functions.coalesce(exact, functions.lit(False))
        completed = functions.struct(
            functions.lit(True).alias(EXACT),
            native.cast(result).alias(NATIVE),
            *(
                functions.lit(None).cast(dtype(type)).alias(name)
                for name, type in program.schema.items()
            ),
        )
        pending = functions.struct(
            functions.lit(False).alias(EXACT),
            functions.lit(None).cast(result).alias(NATIVE),
            *(self.column(Col(name, type)).alias(name) for name, type in program.schema.items()),
        )
        function = functions.arrow_udf(fallback_patch(program), result)  # pyright: ignore[reportCallIssue, reportArgumentType]
        if not program.deterministic:
            function = function.asNondeterministic()
        return function(functions.when(flag, completed).otherwise(pending))

    def evaluate(
        self, exprs: Sequence[Column], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        self.check_names(schema)
        struct = StructType([StructField(name, dtype(type)) for name, type in schema.items()])
        columns = transpose_rows(rows, len(schema))
        table = pa.table(
            {
                name: pa.array(values, type=arrow.dtype(type))
                for (name, type), values in zip(schema.items(), columns, strict=True)
            }
        )
        frame = (
            self.session().createDataFrame(table, struct)
            if schema
            else self.session().range(len(rows)).drop("id")
        )
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        collected = self.execute(lambda: frame.select(*aliased).collect())
        return [[row[index] for row in collected] for index in range(len(exprs))]
