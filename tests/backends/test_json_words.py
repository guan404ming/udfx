import json

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.semantics.text import whitespace
from udfx.testing import check_function
from udfx.types import STR


@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("copied", [False, True])
def test_json_words_skip_element_encoding(ensure_ascii: bool, reverse: bool, copied: bool) -> None:
    def packed(text: str) -> str:
        words = text.rsplit() if reverse else text.split()
        return json.dumps([word for word in words] if copied else words, ensure_ascii=ensure_ascii)

    backend = PolarsBackend()
    schema = {"text": STR.or_none()}
    rows = [
        (None,),
        ("",),
        (whitespace(),),
        ("  alpha\tbeta\n gamma  ",),
        ('quote" slash\\ zero\x00 delete\x7f',),
        (r"\uABCD \\uFFFF",),
        ("中文 😀 café",),
        *((f"left{character}right",) for character in whitespace()),
    ]
    assert check_function(packed, schema, backend, rows) == []
    lowered = backend.lower_program(trace(packed, schema))
    assert ".list.eval(" not in str(lowered.expr)
    assert lowered.exact is not None
    flags = backend.evaluate([lowered.exact], schema, rows)[0]
    assert flags[1:6] == [True] * 5
    assert flags[6] is (not ensure_ascii)
    assert all(flags[7:])


def test_json_word_mapping_keeps_element_encoding() -> None:
    def packed(text: str) -> str:
        return json.dumps([word + " value" for word in text.split()])

    backend = PolarsBackend()
    schema = {"text": STR}
    assert check_function(packed, schema, backend, [("a b",), ("",), ("é 😀",)]) == []
    assert ".list.eval(" in str(backend.lower_program(trace(packed, schema)).expr)
