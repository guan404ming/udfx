from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import count
from typing import ClassVar, cast

import pyarrow as pa
from datafusion import Expr, SessionContext, functions, udf

from udfx.backends import arrow
from udfx.backends.arrow import dtype
from udfx.backends.base import RESERVED, Backend, Row, RuleSet
from udfx.backends.datafusion.rules import RULES, column, typed_literal
from udfx.ir import Col, Lit
from udfx.program import Program
from udfx.types import BOOL, Type

OUT = f"{RESERVED}out"
ROW = f"{RESERVED}row"
FALLBACKS = count()


class DataFusionBackend(Backend[Expr]):
    name: ClassVar[str] = "datafusion"
    rules: ClassVar[RuleSet[Expr]] = RULES

    def __init__(self, context: SessionContext | None = None) -> None:
        self._context = context

    @property
    def context(self) -> SessionContext:
        if self._context is None:
            self._context = SessionContext()
        return self._context

    def column(self, node: Col) -> Expr:
        return column(node.name)

    def literal(self, node: Lit) -> Expr:
        return typed_literal(node.value, node.type)

    def select(self, condition: Expr, then: Expr, orelse: Expr) -> Expr:
        return functions.when(condition, then).otherwise(orelse)

    def both(self, left: Expr, right: Expr) -> Expr:
        return left & right

    def negate(self, value: Expr) -> Expr:
        return ~value

    def fallback(self, native: Expr, exact: Expr, program: Program) -> Expr:
        result = dtype(program.result)

        def run(flags: pa.Array, *columns: pa.Array) -> pa.Array:
            return arrow.fallback(self, program, flags, columns, result)

        slow = udf(
            run,
            [dtype(BOOL), *(dtype(type) for type in program.schema.values())],
            result,
            "volatile",
            f"udfx fallback {next(FALLBACKS)}",
        )
        return functions.when(exact, native).otherwise(
            slow(exact, *(column(name) for name in program.schema))
        )

    def evaluate(
        self, exprs: Sequence[Expr], schema: Mapping[str, Type], rows: Sequence[Row]
    ) -> list[list[object]]:
        self.check_names(schema)
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
        aliased = [expr.alias(name) for expr, name in zip(exprs, names, strict=True)]
        frame = self.context.from_arrow(table)
        outputs = self.execute(
            lambda: frame.select(column(ROW), *aliased).sort(column(ROW)).to_arrow_table()
        )
        return [cast("list[object]", outputs.column(name).to_pylist()) for name in names]
