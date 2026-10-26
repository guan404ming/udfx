from typing import Any

import datafusion
import pyarrow
from datafusion import functions

from bench.callables import batched
from bench.data import CATEGORY, JOIN_KEY, LOOKUP_KEY, POSITION, Input
from bench.engines.base import Adapter
from udfx.backends import arrow
from udfx.backends.datafusion import DataFusionBackend
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
            "immutable",
            "python_arrow",
        )
        return {
            self.primary: implementation(*(datafusion.col(name) for name in self.workload.schema))
        }

    def execute(self, expr: Any, mode: str) -> pyarrow.Table:
        frame = self.context.read_parquet(str(self.data.path)) if mode == "query" else self.frame
        if mode == "query" and self.data.lookup_path is not None:
            frame = frame.join(
                self.context.read_parquet(str(self.data.lookup_path)),
                left_on=JOIN_KEY,
                right_on=LOOKUP_KEY,
                how="left",
            )
        grouped = mode == "query" and (
            self.data.lookup is not None or CATEGORY in self.data.table.column_names
        )
        if grouped:
            projected = frame.select(datafusion.col(CATEGORY), expr.alias("value"))
            return (
                projected.filter(datafusion.col("value").is_not_null())
                .aggregate(
                    [datafusion.col(CATEGORY), datafusion.col("value")],
                    [functions.count(datafusion.lit(1)).alias("count")],
                )
                .to_arrow_table()
            )
        return frame.select(datafusion.col(POSITION), expr.alias("value")).to_arrow_table()
