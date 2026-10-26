from typing import Any

import polars
import pyarrow

from bench.data import CATEGORY, JOIN_KEY, LOOKUP_KEY, POSITION, Input
from bench.engines.base import Adapter
from udfx.backends.polars import PolarsBackend
from udfx.backends.polars.dtypes import dtype
from workloads.models import Workload


class Engine(Adapter):
    primary = "python_scalar"

    def __init__(self, workload: Workload, data: Input, threads: int) -> None:
        super().__init__(workload, data, threads)
        self.backend = PolarsBackend()
        frame = polars.from_arrow(data.table)
        assert isinstance(frame, polars.DataFrame)
        self.frame = frame

    def python(self) -> dict[str, Any]:
        function = self.workload.function
        assert function is not None
        names = list(self.workload.schema)
        if len(names) == 1:
            expr = polars.col(names[0]).map_elements(
                function,
                return_dtype=dtype(self.workload.result),
                skip_nulls=False,
            )
        else:
            expr = polars.struct(names).map_elements(
                lambda row: function(*(row[name] for name in names)),
                return_dtype=dtype(self.workload.result),
            )
        return {self.primary: expr}

    def execute(self, expr: polars.Expr, mode: str) -> pyarrow.Table:
        frame = polars.scan_parquet(self.data.path) if mode == "query" else self.frame.lazy()
        if mode == "query" and self.data.lookup_path is not None:
            frame = frame.join(
                polars.scan_parquet(self.data.lookup_path),
                left_on=JOIN_KEY,
                right_on=LOOKUP_KEY,
                how="left",
            )
        grouped = mode == "query" and (
            self.data.lookup is not None or CATEGORY in self.data.table.column_names
        )
        if grouped:
            result = (
                frame.select(CATEGORY, expr.alias("value"))
                .filter(polars.col("value").is_not_null())
                .group_by(CATEGORY, "value")
                .len(name="count")
            )
        else:
            result = frame.select(POSITION, expr.alias("value"))
        return result.collect().to_arrow()

    def settings(self) -> dict:
        return {**super().settings(), "threads": polars.thread_pool_size(), "execution": "lazy"}
