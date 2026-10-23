import pytest

from bench.speed.data import BATCH_SIZE, dataset
from tests.corpus.cases import CASES, Case
from udfx.frontend import trace


@pytest.mark.parametrize(
    "case", [case for case in CASES if case.expect == "native"], ids=lambda case: case.name
)
def test_dataset(case: Case) -> None:
    program = trace(case.function, case.schema)
    table, expected = dataset(case, program, 200)
    assert table.num_rows == len(expected) == 200
    assert table.column_names == list(case.schema)
    rows = [tuple(row.values()) for row in table.to_pylist()]
    assert program.run(rows) == expected


def test_dataset_splits_engine_batches() -> None:
    case = next(case for case in CASES if case.expect == "native")
    table, _ = dataset(case, trace(case.function, case.schema), BATCH_SIZE + 1)
    assert [len(batch) for batch in table.to_batches()] == [BATCH_SIZE, 1]
