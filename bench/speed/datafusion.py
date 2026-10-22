from collections.abc import Sequence

import datafusion
import pyarrow

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
from udfx.backends.base import RESERVED
from udfx.backends.datafusion import DataFusionBackend
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
    context = datafusion.SessionContext()
    frame = context.from_arrow(table)
    python = datafusion.udf(
        batched(caller(program), arrow.dtype(program.result), len(program.schema)),
        [arrow.dtype(type) for type in program.schema.values()],
        arrow.dtype(program.result),
        "volatile",
        "python_arrow",
    )
    arguments = [datafusion.col(column) for column in program.schema]
    backend = DataFusionBackend(context)
    native = backend.build(program)
    expression = fingerprint(backend, program)
    position = f"{RESERVED}position"
    checked = context.from_arrow(
        table.append_column(position, pyarrow.array(range(table.num_rows), pyarrow.int64()))
    )

    def check(expr: datafusion.Expr) -> None:
        output = checked.select(datafusion.col(position), expr).to_arrow_table()
        ordered = sorted(
            zip(output.column(0).to_pylist(), output.column(1).to_pylist(), strict=True)
        )
        verify("datafusion", name, [value for _, value in ordered], expected)

    def python_time() -> tuple[float, str]:
        check(python(*arguments))
        return timed(lambda: frame.select(python(*arguments)).to_arrow_table(), repeats), "arrow"

    check(native)
    baseline, form = baselines("datafusion", python_time)
    native_time = timed(lambda: frame.select(native).to_arrow_table(), repeats)
    return [
        Measurement(
            name,
            "datafusion",
            baseline,
            native_time,
            form,
            fallback_in_plan(native, backend.name),
            expression,
            pending_share(
                backend,
                program,
                lambda expr: frame.select(expr).to_arrow_table().column(0).to_pylist(),
            ),
        )
    ]
