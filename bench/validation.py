import pyarrow

from bench.data import Input, expected_query
from bench.distributions import verify_distribution
from bench.inspection import verify


def validate(actual: pyarrow.Table, data: Input, mode: str, label: str) -> None:
    expected = (
        expected_query(data)
        if mode == "query"
        else [[index, value] for index, value in enumerate(data.expected)]
    )
    rows = [list(row.values()) for row in actual.to_pylist()]
    if not data.metadata.get("deterministic", True):
        verify_distribution(rows, expected, label)
        return
    verify(label, mode, sorted(rows, key=repr), sorted(expected, key=repr))
