from collections.abc import Callable

import pyarrow as pa
from pyspark.sql import Column, functions

from udfx.backends.base import RESERVED, Backend
from udfx.backends.rules import Lowered
from udfx.backends.spark.dtypes import dtype
from udfx.ir import Col
from udfx.program import Program

EXACT = f"{RESERVED}exact"
NATIVE = f"{RESERVED}native"


def packet(backend: Backend[Column], value: Lowered[Column], program: Program) -> Column:
    result = dtype(program.result)
    completed = functions.struct(
        functions.lit(True).alias(EXACT),
        value.expr.cast(result).alias(NATIVE),
        *(
            functions.lit(None).cast(dtype(type)).alias(name)
            for name, type in program.schema.items()
        ),
    )
    exact = backend.exact(value)
    if exact is None:
        return completed
    pending = functions.struct(
        functions.lit(False).alias(EXACT),
        functions.lit(None).cast(result).alias(NATIVE),
        *(backend.column(Col(name, type)).alias(name) for name, type in program.schema.items()),
    )
    return functions.when(functions.coalesce(exact, functions.lit(False)), completed).otherwise(
        pending
    )


def finish(packet: Column, program: Program, patch: Callable[[pa.Array], pa.Array]) -> Column:
    function = functions.arrow_udf(patch, dtype(program.result))  # pyright: ignore[reportCallIssue, reportArgumentType]
    if not program.deterministic:
        function = function.asNondeterministic()
    return function(packet)
