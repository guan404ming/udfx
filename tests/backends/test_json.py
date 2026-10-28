import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("ensure_ascii", [False, True])
def test_json_list_serialization(backend: Backend[Any], ensure_ascii: bool) -> None:
    def packed(words: list[str | None] | None) -> str:
        return json.dumps(words, ensure_ascii=ensure_ascii)

    schema = {"words": list_of(STR.or_none()).or_none()}
    rows = [
        (None,),
        ([],),
        ([None, "null", ""],),
        (['a"b', "a\\b", "\x00\b\f\n\r\t\x1f\x7f"],),
        (["中文", "😀", "é\u2028"],),
        ([r"\uABCD", r"\\uFFFF", 'a",', "b"],),
        (["AbCDé漢😀\x7f", "\uffff", "\U0010ffff"],),
        (["A" * 20_000 + "é", "\\" * 2_000],),
    ]
    assert check_function(packed, schema, backend, rows) == []
    lowered = backend.lower_program(trace(packed, schema))
    if backend.name == "spark":
        assert lowered.exact is None
    elif lowered.exact is not None:
        flags = backend.evaluate([lowered.exact], schema, rows)[0]
        assert flags[:4] == [True] * 4
        assert flags[4] is (not ensure_ascii)


@pytest.mark.parametrize("backend", backend_params())
def test_json_nullable_nested_lists(backend: Backend[Any]) -> None:
    def packed(words: list[list[str | None] | None]) -> str:
        return json.dumps(words)

    schema = {"words": list_of(list_of(STR.or_none()).or_none())}
    rows = [([],), ([None, [], ["a", None], ["é"]],)]
    assert check_function(packed, schema, backend, rows) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda text: json.dumps(obj=text),
        lambda text: json.dumps(text, indent=None, separators=None, sort_keys=False),
        lambda text: json.dumps(text, ensure_ascii=False),
    ],
)
@pytest.mark.parametrize("backend", backend_params())
def test_json_call_options(function: Callable[..., object], backend: Backend[Any]) -> None:
    assert check_function(function, {"text": STR.or_none()}, backend, [(None,), ('"é',)]) == []


@pytest.mark.parametrize(
    "function",
    [
        lambda text: json.dumps(text, indent=2),
        lambda text: json.dumps(text, separators=(",", ":")),
        lambda text: json.dumps(text, default=str),
        lambda text: json.dumps(text, ensure_ascii=bool(text)),
        lambda text: json.dumps({"value": text}),
    ],
)
def test_json_unsupported_options(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"text": STR})
