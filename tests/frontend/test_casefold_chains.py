from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from bench.inspection import fallback_in_plan
from udfx import compile
from udfx.backends.base import Backend
from udfx.backends.spark import SparkBackend
from udfx.frontend import trace
from udfx.ir import Guarded
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda value: value.title().swapcase().capitalize().casefold(),
        lambda value: value.upper().casefold(),
        lambda value: value.casefold().swapcase().title().casefold(),
        lambda value: value.lower().casefold(),
        lambda value: value.casefold().casefold(),
    ],
)
def test_casefold_chain_preserves_unicode(
    backend: Backend[Any], function: Callable[[str], str]
) -> None:
    schema = {"value": STR.or_none()}
    assert not isinstance(trace(function, schema).value, Guarded)
    points = ("", "\u0131", "İ", "ß", "ẞ", "Σ", "ς", "ͅ", "ΐ", "ǅ", "ᾀ", "Ꭰꭰ", "ɤ", "\u1c89")
    rows = [(None,)] + [
        (context.format(point),)
        for point in points
        for context in ("{}", "A{}b", "\u0131{}\u0131", "AΣ{}", "{}Σa", "İ{}ͅ")
    ]
    assert check_function(function, schema, backend, rows) == []
    if isinstance(backend, SparkBackend):
        translation = compile(function, schema, backend)
        assert translation.ok, translation.unsupported
        assert not fallback_in_plan(translation.expr, "spark")
