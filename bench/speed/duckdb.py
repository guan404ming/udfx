from collections.abc import Callable, Sequence
from typing import Any, cast

import duckdb
import pyarrow
from duckdb.func import ARROW, NATIVE, SPECIAL

from bench.speed.data import batched, caller
from bench.speed.fingerprint import fingerprint
from bench.speed.timing import (
    Baseline,
    Measurement,
    fallback_in_plan,
    measured,
    pending_share,
    timed,
    verify,
)
from udfx.backends import arrow
from udfx.backends.duckdb import DuckDBBackend
from udfx.backends.duckdb.dtypes import dtype
from udfx.program import Program


def measure(
    name: str,
    program: Program,
    table: pyarrow.Table,
    expected: Sequence[object],
    repeats: int,
    *,
    baselines: Baseline = measured,
) -> list[Measurement]:
    function = caller(program)
    parameters = [dtype(type) for type in program.schema.values()]
    returns = dtype(program.result)
    with duckdb.connect() as connection:
        forms = {
            "native": function,
            "arrow": batched(function, arrow.dtype(program.result), len(parameters)),
        }
        relation = connection.from_arrow(table)
        arguments = [duckdb.ColumnExpression(column) for column in program.schema]
        backend = DuckDBBackend(connection)
        native = backend.build(program)
        expression = fingerprint(backend, program)
        verify(
            "duckdb", name, relation.select(native).to_arrow_table().column(0).to_pylist(), expected
        )

        def python_time() -> tuple[float, str]:
            timings: list[tuple[float, str]] = []
            for form, implementation in forms.items():
                connection.create_function(
                    f"python_{form}",
                    cast(Callable[..., Any], implementation),
                    [*parameters],
                    returns,
                    type=ARROW if form == "arrow" else NATIVE,
                    null_handling=SPECIAL,
                )
                python = duckdb.FunctionExpression(f"python_{form}", *arguments)
                verify(
                    f"duckdb {form}",
                    name,
                    relation.select(python).to_arrow_table().column(0).to_pylist(),
                    expected,
                )
                timings.append(
                    (
                        timed(
                            lambda python=python: relation.select(python).to_arrow_table(), repeats
                        ),
                        form,
                    )
                )
            return min(timings)

        baseline, form = baselines("duckdb", python_time)
        native_time = timed(lambda: relation.select(native).to_arrow_table(), repeats)
        fallback = fallback_in_plan(native, backend.name)
        pending = pending_share(
            backend,
            program,
            lambda expr: relation.select(expr).to_arrow_table().column(0).to_pylist(),
        )
    return [Measurement(name, "duckdb", baseline, native_time, form, fallback, expression, pending)]
