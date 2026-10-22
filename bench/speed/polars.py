from collections.abc import Sequence
from functools import partial

import polars
import pyarrow

from bench.speed.data import caller, row_function, row_type
from bench.speed.fingerprint import fingerprint
from bench.speed.timing import Measurement, fallback_in_plan, timed, verify
from udfx.backends.polars import PolarsBackend
from udfx.backends.polars.dtypes import dtype
from udfx.program import Program
from udfx.types import Kind


def measure(
    name: str, program: Program, table: pyarrow.Table, expected: Sequence[object], repeats: int
) -> list[Measurement]:
    frame = polars.from_arrow(table)
    assert isinstance(frame, polars.DataFrame)
    function = row_function(program) if program.row_mode else caller(program)
    names = list(program.schema)
    form = "map_elements"
    if program.row_mode and row_type(program) is tuple:
        python = partial(frame.select(names).map_rows, function, return_dtype=dtype(program.result))
        form = "map_rows"
    elif program.row_mode:
        python = polars.struct(names).map_elements(function, return_dtype=dtype(program.result))
    elif len(names) == 1 and all(type.kind is not Kind.LIST for type in program.schema.values()):
        python = polars.col(names[0]).map_elements(
            function, return_dtype=dtype(program.result), skip_nulls=False
        )
    else:
        python = polars.struct(names).map_elements(
            lambda row: function(*(row[name] for name in names)),
            return_dtype=dtype(program.result),
        )
    backend = PolarsBackend()
    native = backend.build(program)
    expression = fingerprint(backend, program)
    fallback = fallback_in_plan(backend, program)
    measurements: list[Measurement] = []
    for engine, execute in (
        ("polars eager", lambda expr: frame.with_columns(expr.alias("out")).select("out")),
        (
            "polars lazy",
            lambda expr: frame.lazy().with_columns(expr.alias("out")).select("out").collect(),
        ),
    ):
        run_python = partial(execute, python) if isinstance(python, polars.Expr) else python
        verify(engine, name, execute(native)["out"].to_list(), expected)
        verify(engine, name, run_python().to_series().to_list(), expected)
        baseline = timed(run_python, repeats)
        native_time = timed(partial(execute, native), repeats)
        measurements.append(
            Measurement(name, engine, baseline, native_time, form, fallback, expression)
        )
    return measurements
