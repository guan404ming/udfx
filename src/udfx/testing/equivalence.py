from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pickle import dumps
from typing import Any

from udfx.api import Translation, compile
from udfx.backends.base import Backend, Row
from udfx.errors import Unsupported
from udfx.ir import TRUE, Call, Col, Lit
from udfx.ops import Fault, Op
from udfx.testing.allocation import allocation_batches
from udfx.testing.observation import record
from udfx.testing.outcome import Outcome, Raised, Returned, run, same
from udfx.types import BOOL, Type


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
    return check_translation(compile(function, schema, backend), backend, rows)


def check_translation(
    translation: Translation[Any], backend: Backend[Any], rows: Sequence[Row]
) -> list[Mismatch]:
    return check_translation_batches(translation, backend, [rows])


def check_translation_batches(
    translation: Translation[Any], backend: Backend[Any], batches: Sequence[Sequence[Row]]
) -> list[Mismatch]:
    if translation.unsupported is not None or translation.program is None:
        raise translation.unsupported or Unsupported("translation has no program")
    function, schema, expr = (
        translation.program.function,
        translation.program.schema,
        translation.expr,
    )
    expected = [
        [
            run(function, translation.program.arguments(row), translation.program.result)
            for row in rows
        ]
        for rows in batches
    ]
    returned = [
        (row, out)
        for rows, outcomes in zip(batches, expected, strict=True)
        for row, out in zip(rows, outcomes, strict=True)
        if isinstance(out, Returned)
    ]
    (actual,) = backend.evaluate([expr], schema, [row for row, _ in returned])
    assert len(actual) == len(returned), (
        f"{backend.name} returned {len(actual)} rows for {len(returned)} input rows"
    )
    mismatches = [
        Mismatch(row, out, repr(value))
        for (row, out), value in zip(returned, actual, strict=True)
        if not same(out.value, value)
    ]
    attempted: dict[bytes, Outcome] = {}

    def attempt(rows: Sequence[Row]) -> Outcome:
        key = dumps(tuple(rows))
        if key not in attempted:
            attempted[key] = _attempt(backend, expr, schema, rows)
        return attempted[key]

    raising = [
        (kind, row)
        for rows, outcomes in zip(batches, expected, strict=True)
        for kind, row in {
            out.kind: row
            for row, out in zip(rows, outcomes, strict=True)
            if isinstance(out, Raised)
        }.items()
    ]
    for kind, row in raising:
        got = attempt([row])
        if not (isinstance(got, Raised) and got.kind is kind):
            mismatches.append(Mismatch(row, Raised(kind), repr(got)))
    rows = [row for batch in batches for row in batch]
    if raising and not isinstance(attempt(rows), Raised):
        mismatches.append(Mismatch(rows[0], Raised(Exception), "batch did not raise"))
    return mismatches


def check_op(
    op: Op,
    backend: Backend[Any],
    rows: Sequence[Row],
    constants: Mapping[int, object] | None = None,
) -> list[Mismatch]:
    return check_op_batches(op, backend, [(rows, constants or {})])


def check_op_batches(
    op: Op,
    backend: Backend[Any],
    batches: Sequence[tuple[Sequence[Row], Mapping[int, object]]],
) -> list[Mismatch]:
    grouped: dict[bytes, tuple[list[Row], Mapping[int, object]]] = {}
    for rows, constants in batches:
        key = dumps(tuple(sorted(constants.items())))
        if key not in grouped:
            grouped[key] = ([], constants)
        grouped[key][0].extend(rows)
    return [
        mismatch
        for group in batched(grouped.values(), 5)
        for mismatch in _check_op_group(op, backend, group)
    ]


def _check_op_group(
    op: Op,
    backend: Backend[Any],
    batches: Sequence[tuple[Sequence[Row], Mapping[int, object]]],
) -> list[Mismatch]:
    prepared = [
        _prepare_op(op, backend, rows, constants, index)
        for index, (rows, constants) in enumerate(batches)
    ]
    schema = {name: type for _, columns, _, _, _ in prepared for name, type in columns.items()}
    padding = [min(inputs, key=_footprint) for inputs, _, _, _, _ in prepared]
    rows = [
        tuple(
            value
            for (inputs, _, _, _, _), pad in zip(prepared, padding, strict=True)
            for value in (*(inputs[index] if index < len(inputs) else pad), index < len(inputs))
        )
        for index in range(max(len(inputs) for inputs, _, _, _, _ in prepared))
    ]
    mismatches: list[Mismatch] = []
    for offset, batch in allocation_batches(op, rows):
        outputs = iter(
            backend.evaluate(
                [expr for _, _, exprs, _, _ in prepared for expr in exprs], schema, batch
            )
        )
        for inputs, _, exprs, faults, exact_count in prepared:
            results = [next(outputs) for _ in exprs]
            mismatches.extend(
                _op_mismatches(
                    op, inputs[offset : offset + len(batch)], faults, exact_count, results
                )
            )
    return mismatches


def _footprint(row: Row) -> int:
    return sum(
        abs(value)
        if isinstance(value, int)
        else len(value)
        if isinstance(value, str | list | tuple)
        else 0
        for value in row
    )


def _prepare_op(
    op: Op,
    backend: Backend[Any],
    rows: Sequence[Row],
    constants: Mapping[int, object],
    batch: int,
) -> tuple[list[Row], dict[str, Type], list[Any], tuple[Fault, ...], int]:
    columns = tuple(
        Col(f"batch{batch}_arg{index}", parameter) for index, parameter in enumerate(op.parameters)
    )
    schema = {column.name: column.type for column in columns}
    present = Col(f"batch{batch}_present", BOOL)
    schema[present.name] = present.type
    arguments = tuple(
        Lit(constants[index], column.type) if index in constants else column
        for index, column in enumerate(columns)
    )
    rows = [tuple(constants.get(index, item) for index, item in enumerate(row)) for row in rows]
    value = backend.lower(Call(op, arguments))
    faults = op.faults(*arguments)
    lowered = [backend.lower(fault.when) for fault in faults]
    exact_parts = [backend.exact(value), *(backend.exact(item) for item in lowered)]
    exact = [part if part is not None else backend.literal(TRUE) for part in exact_parts]

    exprs = [backend.column(columns[0]), value.expr, *exact, *(item.expr for item in lowered)]
    types = [columns[0].type, op.result, *([BOOL] * (len(exprs) - 2))]
    exprs = [
        backend.select(backend.column(present), expr, backend.literal(Lit(None, type.or_none())))
        for expr, type in zip(exprs, types, strict=True)
    ]
    return rows, schema, exprs, faults, len(exact)


def _op_mismatches(
    op: Op,
    rows: Sequence[Row],
    faults: Sequence[Fault],
    exact_count: int,
    outputs: Sequence[Sequence[object]],
) -> list[Mismatch]:
    values = outputs[1]
    exacts = outputs[2 : 2 + exact_count]
    hits = outputs[2 + exact_count :]
    mismatches: list[Mismatch] = []
    for index, row in enumerate(rows):
        flags_here = [flags[index] for flags in (*exacts, *hits)]
        if any(flag not in (True, False, None) for flag in flags_here):
            mismatches.append(Mismatch(row, run(op.reference, row), f"predicates {flags_here}"))
            continue
        if not all(flags[index] is True for flags in exacts):
            record(row, None)
            continue
        expected = run(op.reference, row)
        record(row, expected)
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
