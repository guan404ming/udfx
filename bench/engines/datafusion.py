from typing import Any

import datafusion
import pyarrow
from datafusion import functions

from bench.callables import batched
from bench.data import CATEGORY, POSITION, Input
from bench.engines.base import Adapter
from udfx.backends import arrow
from udfx.backends.engines.datafusion import DataFusionBackend
from udfx.backends.engines.datafusion.rules import column
from workloads.models import Workload


class Engine(Adapter):
    primary = "python_arrow"

    def __init__(self, workload: Workload, data: Input, threads: int) -> None:
        super().__init__(workload, data, threads)
        self.context = datafusion.SessionContext(
            datafusion.SessionConfig().with_target_partitions(threads)
        )
        self.backend = DataFusionBackend(self.context)
        self.frame = self.context.from_arrow(data.table)

    def python(self) -> dict[str, Any]:
        function = self.workload.function
        assert function is not None
        implementation = datafusion.udf(
            batched(function, arrow.dtype(self.workload.result), len(self.workload.schema)),
            [arrow.dtype(type) for type in self.workload.schema.values()],
            arrow.dtype(self.workload.result),
            "immutable" if self.workload.deterministic else "volatile",
            "python_arrow",
        )
        return {self.primary: implementation(*(column(name) for name in self.workload.schema))}

    def execute(self, expr: Any, mode: str) -> pyarrow.Table:
        frame = self.context.read_parquet(str(self.data.path)) if mode == "query" else self.frame
        grouped = mode == "query" and CATEGORY in self.data.table.column_names
        if grouped:
            projected = frame.select(column(CATEGORY), expr.alias("value"))
            return (
                projected.filter(column("value").is_not_null())
                .aggregate(
                    [column(CATEGORY), column("value")],
                    [functions.count(datafusion.lit(1)).alias("count")],
                )
                .to_arrow_table()
            )
        return frame.select(column(POSITION), expr.alias("value")).to_arrow_table()
