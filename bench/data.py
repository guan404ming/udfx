import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow
from pyarrow import parquet

from udfx.backends import arrow
from udfx.backends.base import RESERVED
from udfx.testing.outcome import Raised, run
from workloads.models import Workload
from workloads.storage import digest

POSITION = f"{RESERVED}position"
CATEGORY = f"{RESERVED}category"


@dataclass(frozen=True)
class Input:
    table: pyarrow.Table
    path: Path
    expected: list[object]
    metadata: dict


def prepare(workload: Workload, directory: Path, rows: int | None) -> Input:
    if workload.function is None:
        raise ValueError(workload.error or "Missing Python function")
    source = parquet.read_table(workload.table)
    if rows is not None:
        source = source.slice(0, rows)
    if not source.num_rows:
        raise ValueError("The workload has no input rows")
    columns = {}
    for name, type in workload.schema.items():
        column = (
            source[workload.columns[name]]
            if name in workload.columns
            else pyarrow.array([workload.constants[name]] * source.num_rows, arrow.dtype(type))
        )
        columns[name] = column.cast(arrow.dtype(type))
    table = pyarrow.table(columns)
    expected = []
    for index, row in enumerate(
        zip(*(column.to_pylist() for column in table.columns), strict=True)
    ):
        outcome = run(workload.function, row, workload.result)
        if isinstance(outcome, Raised):
            raise ValueError(f"Python oracle raised {outcome.kind.__name__} at input row {index}")
        expected.append(outcome.value)
    table = table.append_column(POSITION, pyarrow.array(range(table.num_rows), pyarrow.int64()))
    if workload.category is not None:
        table = table.append_column(CATEGORY, source[workload.category])
    metadata = {
        "source_sha256": digest(workload.table),
        "columns": dict(workload.columns),
        "constants": dict(workload.constants),
        "category": workload.category,
        "rows": table.num_rows,
        "row_selection": "source prefix" if rows is not None else "all source rows",
        "schema": {name: str(type) for name, type in workload.schema.items()},
        "deterministic": workload.deterministic,
    }
    directory.mkdir(parents=True, exist_ok=True)
    identifier = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = directory / f"{identifier}.parquet"
    parquet.write_table(table, path)
    metadata["input_sha256"] = digest(path)
    return Input(table, path, expected, metadata)


def expected_query(data: Input) -> list[list[object]]:
    if CATEGORY not in data.table.column_names:
        return [[index, value] for index, value in enumerate(data.expected)]
    categories = data.table[CATEGORY].to_pylist()
    counts: dict[tuple[object, object], int] = {}
    for category, value in zip(categories, data.expected, strict=True):
        if value is not None:
            key = (category, value)
            counts[key] = counts.get(key, 0) + 1
    return [[category, value, count] for (category, value), count in counts.items()]
