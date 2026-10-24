from datetime import datetime
from typing import Any

import pytest
from conftest import backend_params

from bench.speed.timing import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("format", ["%Y", "%Y-%m", "%Y-%m-%d"])
def test_timestamp_prefix_skips_field_extraction(backend: Backend[Any], format: str) -> None:
    def parse(text: str) -> datetime:
        return datetime.strptime(text, format)

    schema = {"text": STR}
    translation = compile(parse, schema, backend)
    assert translation.ok, translation.unsupported
    if isinstance(backend, SparkBackend):
        assert "regexp_extract(" not in str(translation.expr)
        assert not fallback_in_plan(translation.expr, "spark")
    values = [
        datetime(year, month, 1).strftime(format)
        for year in (1000, 1900, 2000, 2024, 9999)
        for month in (1, 2, 12)
    ]
    values.extend(
        [
            "",
            "0000",
            "24",
            "24-01",
            "24-01-01",
            "10000-01-01",
            "2024-2-9",
            "2024-02-29",
            "1900-02-29",
            "\uff12\uff10\uff12\uff14-01-01",
        ]
    )
    assert check_function(parse, schema, backend, [(text,) for text in values]) == []
