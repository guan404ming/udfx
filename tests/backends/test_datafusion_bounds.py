import pytest

from udfx.backends.datafusion import DataFusionBackend
from udfx.frontend import trace
from udfx.types import STR, Type, list_of


@pytest.mark.parametrize("container", [STR, list_of(STR)])
def test_large_arrow_containers_keep_length_overflow_guard(container: Type) -> None:
    program = trace(lambda value: len(value) * 4, {"value": container})
    lowered = DataFusionBackend().lower(program.value)
    assert lowered.overflow is not None
