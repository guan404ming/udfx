import heapq
from collections import Counter
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.testing import check_function
from udfx.types import INT, STR


@pytest.mark.parametrize("backend", backend_params())
def test_frequency_ranking(backend: Backend[Any]) -> None:
    def frequent(text: str, count: int) -> str:
        counts: Any = Counter(word.lower() for word in text.split())
        return " ".join(heapq.nlargest(count, counts, key=counts.get))

    rows = [
        (text, count)
        for text in ("", "B a b A c", "z a z b a b", "İ i 中文 中文 😀", "x\x00y x\x00y")
        for count in (-3, 0, 1, 2, 10)
    ]
    assert check_function(frequent, {"text": STR, "count": INT}, backend, rows) == []
