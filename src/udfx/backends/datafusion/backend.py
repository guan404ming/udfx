from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import count
from typing import ClassVar, cast

import pyarrow as pa
from datafusion import Expr, SessionContext, col, functions, udf

from udfx.backends.arrow import dtype
from udfx.backends.base import Backend, Row, RuleSet
from udfx.backends.datafusion.rules import RULES, typed
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import BOOL, Type

OUT = "udfx out"
ROW = "udfx row"
FALLBACKS = count()


class DataFusionBackend(Backend[Expr]):
    name: ClassVar[str] = "datafusion"
    rules: ClassVar[RuleSet[Expr]] = RULES

    def __init__(self, context: SessionContext | None = None) -> None:
        self.context = context or SessionContext()

    def column(self, node: Col) -> Expr:
        return col(node.name)

    def literal(self, node: Lit) -> Expr:
        return typed(node.value, node.type)

    def select(self, condition: Expr, then: Expr, orelse: Expr) -> Expr:
        return functions.when(condition, then).otherwise(orelse)

    def both(self, left: Expr, right: Expr) -> Expr:
        return left & right

    def negate(self, value: Expr) -> Expr:
        return ~value

    def fallback(self, native: Expr, exact: Expr, program: Program) -> Expr:
        result = dtype(program.result)

        def run(flags: pa.Array, *columns: pa.Array) -> pa.Array:
            pending = [index for index, flag in enumerate(flags.to_pylist()) if not flag]
            arguments = [column.to_pylist() for column in columns]
            rows = [tuple(values[index] for values in arguments) for index in pending]
            values: list[object] = [None] * len(flags)
            for index, value in zip(pending, self.patch(program, rows), strict=True):
                values[index] = value
            return pa.array(values, type=result)

        slow = udf(
            run,
            [dtype(BOOL), *(dtype(type) for type in program.schema.values())],
            result,
            "volatile",
            f"udfx fallback {next(FALLBACKS)}",
        )
        return functions.when(exact, native).otherwise(
            slow(exact, *(col(name) for name in program.schema))
        )

    def evaluate(
        self, exprs: Sequence[Expr], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        columns = list(zip(*rows, strict=True)) if rows else [() for _ in schema]
        table = pa.table(
            {
                ROW: pa.array(range(len(rows)), type=pa.int64()),
                **{
                    name: pa.array(list(values), type=dtype(type))
                    for (name, type), values in zip(schema.items(), columns, strict=True)
                },
            }
        )
        names = [f"{OUT} {index}" for index in range(len(exprs))]
        frame = self.context.from_arrow(table).select(
            col(ROW), *(expr.alias(name) for expr, name in zip(exprs, names, strict=True))
        )
        try:
            outputs = frame.sort(col(ROW)).to_arrow_table()
        except Exception as wrapped:
            raise self.unwrap(wrapped) from None
        return [cast("list[object]", outputs.column(name).to_pylist()) for name in names]
