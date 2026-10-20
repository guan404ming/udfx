from udfx.backends.polars import PolarsBackend
from udfx.ir import Lit
from udfx.program import Program
from udfx.types import INT

POLARS = PolarsBackend()


def test_evaluate_keeps_row_count_without_columns() -> None:
    program = Program(lambda: 1, {}, Lit(1, INT))
    assert POLARS.evaluate([POLARS.build(program)], {}, [(), (), ()]) == [[1, 1, 1]]
