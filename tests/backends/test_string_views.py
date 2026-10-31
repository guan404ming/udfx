import pyarrow as pa
import pytest

from udfx.backends.datafusion import DataFusionBackend
from udfx.frontend import trace
from udfx.types import STR, list_of


@pytest.mark.parametrize("string_type", [pa.string(), pa.string_view()])
def test_comprehensions_keep_string_views(string_type: pa.DataType) -> None:
    def stripped(values: list[str]) -> list[str]:
        return [value.strip() for value in values]

    backend = DataFusionBackend()
    rows = [[], ["  alpha  ", " 中文 ", " 😀 "], ["", "\t", "a\x00b"]]
    table = pa.table({"values": pa.array(rows, type=pa.list_(string_type))})
    program = trace(stripped, {"values": list_of(STR)})
    result = backend.context.from_arrow(table).select(backend.build(program).alias("value"))
    output = result.to_arrow_table().column("value")
    assert output.type == pa.list_(pa.string_view())
    assert output.to_pylist() == [stripped(values) for values in rows]
