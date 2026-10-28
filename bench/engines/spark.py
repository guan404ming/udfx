from typing import Any

import pyarrow
from pyspark.sql import SparkSession, functions

from bench.callables import batched
from bench.data import CATEGORY, JOIN_KEY, LOOKUP_KEY, POSITION, Input
from bench.engines.base import Adapter
from udfx.backends import arrow
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.dtypes import dtype
from workloads.models import Workload


class Engine(Adapter):
    primary = "python_scalar"

    def __init__(self, workload: Workload, data: Input, threads: int) -> None:
        super().__init__(workload, data, threads)
        self.session = (
            SparkSession.builder.master(f"local[{threads}]")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.shuffle.partitions", str(threads))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
            .getOrCreate()
            .newSession()
        )
        self.backend = SparkBackend(self.session)
        self.frame = self.session.createDataFrame(data.table).cache()
        self.frame.count()

    def python(self) -> dict[str, Any]:
        function = self.workload.function
        assert function is not None
        arguments = [functions.col(name) for name in self.workload.schema]
        implementations = {
            "python_scalar": functions.udf(function, dtype(self.workload.result), useArrow=False),
            "python_arrow": functions.arrow_udf(
                batched(function, arrow.dtype(self.workload.result), len(self.workload.schema)),
                dtype(self.workload.result),
                functions.ArrowUDFType.SCALAR,
            ),
        }
        return {
            name: (function if self.workload.deterministic else function.asNondeterministic())(
                *arguments
            )
            for name, function in implementations.items()
        }

    def execute(self, expr: Any, mode: str) -> pyarrow.Table:
        frame = self.session.read.parquet(str(self.data.path)) if mode == "query" else self.frame
        if mode == "query" and self.data.lookup_path is not None:
            lookup = self.session.read.parquet(str(self.data.lookup_path))
            frame = frame.join(lookup, frame[JOIN_KEY] == lookup[LOOKUP_KEY], "left")
        grouped = mode == "query" and (
            self.data.lookup is not None or CATEGORY in self.data.table.column_names
        )
        if grouped:
            frame = frame.select(CATEGORY, expr.alias("value")).filter(
                functions.col("value").isNotNull()
            )
            return frame.groupBy(CATEGORY, "value").count().toArrow()
        return frame.select(POSITION, expr.alias("value")).toArrow()

    def settings(self) -> dict:
        return {
            **super().settings(),
            "memory_input": "Spark cached Arrow input",
            "master": self.session.sparkContext.master,
            "spark": dict(self.session.conf.getAll),
        }

    def close(self) -> None:
        self.frame.unpersist(blocking=True)
