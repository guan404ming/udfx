from collections.abc import Callable
from typing import Any, cast

import duckdb
import pyarrow
from duckdb.func import ARROW, NATIVE, SPECIAL

from bench.callables import batched
from bench.data import CATEGORY, JOIN_KEY, LOOKUP_KEY, POSITION, Input
from bench.engines.base import Adapter
from udfx.backends import arrow
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.duckdb.dtypes import dtype
from workloads.models import Workload


class Engine(Adapter):
    primary = "python_scalar"

    def __init__(self, workload: Workload, data: Input, threads: int) -> None:
        super().__init__(workload, data, threads)
        self.connection = duckdb.connect(config={"threads": threads})
        self.backend = DuckDBBackend(self.connection)
        self.frame = self.connection.from_arrow(data.table)

    def python(self) -> dict[str, Any]:
        function = self.workload.function
        assert function is not None
        implementations = {
            "python_scalar": (function, NATIVE),
            "python_arrow": (
                batched(function, arrow.dtype(self.workload.result), len(self.workload.schema)),
                ARROW,
            ),
        }
        expressions = {}
        for name, (implementation, form) in implementations.items():
            self.connection.create_function(
                name,
                cast(Callable[..., Any], implementation),
                [dtype(type) for type in self.workload.schema.values()],
                dtype(self.workload.result),
                type=form,
                null_handling=SPECIAL,
            )
            expressions[name] = duckdb.FunctionExpression(
                name,
                *(duckdb.ColumnExpression(name) for name in self.workload.schema),
            )
        return expressions

    def execute(self, expr: Any, mode: str) -> pyarrow.Table:
        frame = self.connection.read_parquet(str(self.data.path)) if mode == "query" else self.frame
        if mode == "query" and self.data.lookup_path is not None:
            lookup = self.connection.read_parquet(str(self.data.lookup_path))
            frame = frame.join(lookup, f'"{JOIN_KEY}" = "{LOOKUP_KEY}"', how="left")
        grouped = mode == "query" and (
            self.data.lookup is not None or CATEGORY in self.data.table.column_names
        )
        if grouped:
            frame = frame.select(duckdb.ColumnExpression(CATEGORY), expr.alias("value"))
            return (
                frame.filter('"value" IS NOT NULL')
                .aggregate(
                    f'"{CATEGORY}", "value", count(*) AS "count"',
                    f'"{CATEGORY}", "value"',
                )
                .to_arrow_table()
            )
        return frame.select(duckdb.ColumnExpression(POSITION), expr.alias("value")).to_arrow_table()

    def close(self) -> None:
        self.connection.close()
