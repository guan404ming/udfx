from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from udfx.api import compile
from udfx.backends.base import Backend, Row
from udfx.errors import Unsupported
from udfx.ir import TRUE, Call, Col, Lit
from udfx.ops import Op
from udfx.testing.outcome import Outcome, Raised, Returned, run, same
from udfx.types import Type


@dataclass(frozen=True)
class Mismatch:
    row: Row
    expected: Outcome
    actual: str


def check_function(
    function: Callable[..., object],
    schema: Mapping[str, Type],
    backend: Backend[Any],
    rows: Sequence[Row],
) -> list[Mismatch]:
    translation = compile(function, schema, backend)
    if translation.unsupported is not None:
        raise translation.unsupported
    expr = translation.expr
    result = translation.program.result if translation.program else None
    expected = [run(function, row, result) for row in rows]
    returned = [
        (row, out) for row, out in zip(rows, expected, strict=True) if isinstance(out, Returned)
    ]
    (actual,) = backend.evaluate([expr], schema, [row for row, _ in returned])
    mismatches = [
        Mismatch(row, out, repr(value))
        for (row, out), value in zip(returned, actual, strict=True)
        if not same(out.value, value)
    ]
    for row, out in zip(rows, expected, strict=True):
        if isinstance(out, Raised):
            got = _attempt(backend, expr, schema, [row])
            if not (isinstance(got, Raised) and got.kind is out.kind):
                mismatches.append(Mismatch(row, out, repr(got)))
    if len(returned) < len(rows) and not isinstance(_attempt(backend, expr, schema, rows), Raised):
        mismatches.append(Mismatch(rows[0], Raised(Exception), "batch did not raise"))
    return mismatches


def check_op(
    op: Op,
    backend: Backend[Any],
    rows: Sequence[Row],
    constants: Mapping[int, object] | None = None,
) -> list[Mismatch]:
    constants = constants or {}
    columns = tuple(Col(f"arg{index}", param) for index, param in enumerate(op.params))
    schema = {column.name: column.type for column in columns}
    args = tuple(
        Lit(constants[index], column.type) if index in constants else column
        for index, column in enumerate(columns)
    )
    rows = [tuple(constants.get(index, item) for index, item in enumerate(row)) for row in rows]
    value = backend.lower(Call(op, args))
    faults = op.faults(*args)
    lowered = [backend.lower(fault.when) for fault in faults]
    exact_parts = [value.exact, *(item.exact for item in lowered)]
    exact = [part if part is not None else backend.literal(TRUE) for part in exact_parts]

    outputs = backend.evaluate([value.expr, *exact, *(item.expr for item in lowered)], schema, rows)
    values, exacts, hits = outputs[0], outputs[1 : 1 + len(exact)], outputs[1 + len(exact) :]
    mismatches: list[Mismatch] = []
    for index, row in enumerate(rows):
        flags_here = [flags[index] for flags in (*exacts, *hits)]
        if any(flag not in (True, False, None) for flag in flags_here):
            mismatches.append(Mismatch(row, run(op.reference, row), f"predicates {flags_here}"))
            continue
        if not all(flags[index] is True for flags in exacts):
            continue
        expected = run(op.reference, row)
        raised = [
            fault.kind for fault, flags in zip(faults, hits, strict=True) if flags[index] is True
        ]
        if isinstance(expected, Raised):
            if not raised or expected.kind is not raised[0]:
                mismatches.append(Mismatch(row, expected, f"faults {raised}"))
        elif raised:
            mismatches.append(Mismatch(row, expected, f"spurious faults {raised}"))
        elif not same(expected.value, values[index]):
            mismatches.append(Mismatch(row, expected, repr(values[index])))
    return mismatches


def _attempt(
    backend: Backend[Any], expr: object, schema: Mapping[str, Type], rows: Sequence[Row]
) -> Outcome:
    try:
        return Returned(backend.evaluate([expr], schema, rows)[0])
    except Unsupported:
        raise
    except Exception as error:
        return Raised(type(error))
