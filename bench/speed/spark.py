from collections.abc import Callable, Sequence

import pyarrow
from pyspark.sql import Column, DataFrame, functions

from bench.speed.data import batched, caller
from bench.speed.timing import Measurement, fallback_in_plan, timed, verify
from udfx.backends import arrow
from udfx.backends.base import RESERVED
from udfx.backends.spark import SparkBackend
from udfx.backends.spark.dtypes import dtype
from udfx.program import Program


def consume(frame: DataFrame, expr: Column) -> Callable[[], object]:
    return lambda: frame.select(functions.sum(functions.hash(expr))).collect()


def measure(
    name: str, program: Program, table: pyarrow.Table, expected: Sequence[object], repeats: int
) -> list[Measurement]:
    backend = SparkBackend()
    position = f"{RESERVED}position"
    table = table.append_column(position, pyarrow.array(range(table.num_rows), pyarrow.int64()))
    frame = backend.session().createDataFrame(table).cache()
    try:
        frame.count()
        arguments = [functions.col(column) for column in program.schema]
        native = backend.build(program)
        function = caller(program)
        forms = {
            "pickled": functions.udf(function, dtype(program.result), useArrow=False),
            "arrow": functions.arrow_udf(
                batched(function, arrow.dtype(program.result), len(program.schema)),
                dtype(program.result),
                functions.ArrowUDFType.SCALAR,
            ),
        }
        for expr in (native, *(implementation(*arguments) for implementation in forms.values())):
            output = frame.select(functions.col(position), expr).collect()
            verify(
                "spark", name, [row[1] for row in sorted(output, key=lambda row: row[0])], expected
            )
        baseline, form = min(
            (timed(consume(frame, implementation(*arguments)), repeats), form)
            for form, implementation in forms.items()
        )
        native_time = timed(consume(frame, native), repeats)
        return [
            Measurement(
                name, "spark", baseline, native_time, form, fallback_in_plan(backend, program)
            )
        ]
    finally:
        frame.unpersist()
