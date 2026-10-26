import pyarrow

from bench.data import Input, expected_query
from bench.inspection import verify


def validate(actual: pyarrow.Table, data: Input, mode: str, label: str) -> None:
    expected = (
        expected_query(data)
        if mode == "query"
        else [[index, value] for index, value in enumerate(data.expected)]
    )
    rows = [list(row.values()) for row in actual.to_pylist()]
    verify(label, mode, sorted(rows, key=repr), sorted(expected, key=repr))
