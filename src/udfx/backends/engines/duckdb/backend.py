from __future__ import annotations

import itertools
import string
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import ClassVar, cast

import duckdb
import pyarrow as pa
from duckdb import Expression, sqltypes
from duckdb.func import ARROW, SPECIAL

from udfx.backends import arrow
from udfx.backends.base import RESERVED, Backend, Row, output_names, transpose_rows
from udfx.backends.engines.duckdb.connections import CONNECTION, fallback_name
from udfx.backends.engines.duckdb.dtypes import dtype
from udfx.backends.engines.duckdb.rules import RULES, bind, column, constant, function, when
from udfx.backends.faults import NativeFaults, faults_for
from udfx.backends.lowering.context import lowering_context
from udfx.backends.lowering.programs import lower_program
from udfx.backends.lowering.results import FallbackLowering, ResultLowering, fallback_result
from udfx.backends.registration.shared import Lowered, RuleSet
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import Kind, Type

EXACT = f"{RESERVED}exact"
NATIVE = f"{RESERVED}native"
ROW = f"{RESERVED}row"
ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)
ELEMENTS = itertools.count()
DEFAULT_FAULTS = NativeFaults()


class DuckDBBackend(Backend[Expression]):
    name: ClassVar[str] = "duckdb"
    rules: ClassVar[RuleSet[Expression]] = RULES
    binds: ClassVar[bool] = True
    graphs: ClassVar[bool] = True
    choose_graphs: ClassVar[bool] = True

    def __init__(self, connection: duckdb.DuckDBPyConnection | None = None) -> None:
        self._connection = connection

    def lower_program(self, program: Program) -> Lowered[Expression]:
        token = CONNECTION.set(self.connection)
        try:
            return lower_program(self, program, ResultLowering(self, program))
        finally:
            CONNECTION.reset(token)

    def compose(self, value: Lowered[Expression], program: Program) -> Expression:
        value = fallback_result(
            value,
            FallbackLowering(self, program),
            lambda results: lower_program(self, program, results),
        )
        return super().compose(value, program)

    def build_candidate(self, program: Program) -> Expression:
        with lowering_context(binds=self.binds) as context:
            expr = super().build_candidate(program)
        if (
            context.has_bindings
            and program.schema
            and all(type.kind not in {Kind.FLOAT, Kind.LIST} for type in program.schema.values())
        ):
            arguments = ", ".join(
                str(self.column(Col(name, type))) for name, type in program.schema.items()
            )
            return duckdb.SQLExpression(
                f'(SELECT {expr} FROM (SELECT {arguments}) AS "udfx input")'
            )
        return expr

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.default_connection() if self._connection is None else self._connection

    def fold_name(self, name: str) -> str:
        return name.translate(ASCII_LOWER)

    def column(self, node: Col) -> Expression:
        return column(node.name)

    def literal(self, node: Lit) -> Expression:
        return constant(node.value).cast(dtype(node.type))

    def select(self, condition: Expression, then: Expression, orelse: Expression) -> Expression:
        return when(condition, then, orelse)

    def select_many(
        self, cases: Sequence[tuple[Expression, Expression]], orelse: Expression
    ) -> Expression:
        selected = duckdb.CaseExpression(*cases[0])
        for condition, value in cases[1:]:
            selected = selected.when(condition, value)
        return selected.otherwise(orelse)

    def bind(self, value: Expression, body: Callable[[Expression], Expression]) -> Expression:
        return bind(value, body, f"{RESERVED}element {next(ELEMENTS)}")

    def bind_sequence(
        self, value: Expression, steps: Sequence[Callable[[Expression], Expression]]
    ) -> Expression:
        items = function("list_value", value)
        for step in steps:
            items = self.list_transform(items, step)
        return function("list_extract", items, constant(1))

    def select_value(
        self,
        value: Expression,
        cases: Sequence[tuple[Expression, Expression]],
        orelse: Expression,
    ) -> Expression:
        keys = function("list_value", *(key for key, _ in cases))
        values = function("list_value", *(result for _, result in cases))
        selected = function("map_extract_value", function("map", keys, values), value)
        return duckdb.CoalesceOperator(selected, orelse)

    def record(self, values: Sequence[Expression]) -> Expression:
        return function(
            "struct_pack", *(value.alias(str(index)) for index, value in enumerate(values))
        )

    def field(self, value: Expression, index: int) -> Expression:
        return function("struct_extract", value, constant(str(index)))

    def list_value(self, items: Sequence[Expression], type: Type) -> Expression:
        return function("list_value", *items).cast(dtype(type))

    def list_transform(
        self, source: Expression, transform: Callable[[Expression], Expression]
    ) -> Expression:
        name = f"{RESERVED}element {next(ELEMENTS)}"
        return function(
            "list_transform", source, duckdb.LambdaExpression(name, transform(column(name)))
        )

    def list_distinct(
        self, source: Expression, transform: Callable[[Expression], Expression]
    ) -> Expression:
        return self.bind(
            function("list_distinct", source),
            lambda keys: self.bind(
                self.list_transform(keys, transform),
                lambda values: self.list_transform(
                    source,
                    lambda item: function(
                        "list_extract", values, function("list_position", keys, item)
                    ),
                ),
            ),
        )

    def list_filter(
        self, source: Expression, predicate: Callable[[Expression], Expression]
    ) -> Expression:
        name = f"{RESERVED}element {next(ELEMENTS)}"
        return function(
            "list_filter", source, duckdb.LambdaExpression(name, predicate(column(name)))
        )

    @property
    def native_faults(self) -> NativeFaults:
        return DEFAULT_FAULTS if self._connection is None else faults_for(self._connection)

    def native_raise(self, program: Program) -> Expression:
        packed = (
            function(
                "struct_pack",
                *(column(name).alias(str(index)) for index, name in enumerate(program.schema)),
            )
            if program.schema
            else constant({})
        )
        encoded = function("hex", function("encode", function("to_json", packed)))
        message = function("concat", constant(self.native_faults.marker(program)), encoded)
        return function("error", message).cast(dtype(program.result))

    def fallback(self, native: Expression, exact: Expression, program: Program) -> Expression:
        connection = self.connection
        name = fallback_name(program, connection)
        fields = {
            EXACT: sqltypes.BOOLEAN,
            NATIVE: dtype(program.result),
            **{field: dtype(type) for field, type in program.schema.items()},
        }

        def patch(packed: pa.ChunkedArray) -> pa.Array:
            batch = cast(pa.StructArray, packed.combine_chunks())
            columns = [batch.field(field) for field in program.schema]
            return arrow.fallback(
                DuckDBBackend(), program, batch.field(EXACT), batch.field(NATIVE), columns
            )

        with suppress(duckdb.NotImplementedException):
            connection.create_function(
                name,
                patch,
                [duckdb.struct_type(fields)],
                dtype(program.result),
                type=ARROW,
                null_handling=SPECIAL,
                side_effects=not program.deterministic,
            )
        packed = function(
            "struct_pack",
            constant(False).alias(EXACT),
            constant(None).cast(dtype(program.result)).alias(NATIVE),
            *(column(field).alias(field) for field in program.schema),
        )
        return when(exact, native, function(name, packed))

    def evaluate(
        self,
        exprs: Sequence[Expression],
        schema: Mapping[str, Type],
        rows: Sequence[Row],
        reduced: bool = False,
    ) -> list[list[object]]:
        self.check_names(schema)
        columns = transpose_rows(rows, len(schema))
        table = pa.table(
            {
                ROW: pa.array(range(len(rows)), pa.int64()),
                **{
                    name: pa.array(list(values))
                    for name, values in zip(schema, columns, strict=True)
                },
            }
        )
        typed = self.connection.from_arrow(table).select(
            column(ROW),
            *(column(name).cast(dtype(type)).alias(name) for name, type in schema.items()),
        )
        names = output_names(len(exprs))
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        relation = typed.aggregate(aliased) if reduced else typed.select(*aliased)
        outputs = self.execute(relation.fetchall)
        return [list(values) for values in zip(*outputs, strict=True)] or [[] for _ in exprs]
