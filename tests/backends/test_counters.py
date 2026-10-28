import heapq
from collections import Counter
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.lowering_context import lowering_context
from udfx.frontend import trace
from udfx.ir import Call, Col, lit
from udfx.ops import OpKey, apply
from udfx.testing import check_function
from udfx.types import INT, STR, Kind, list_of


def test_frequency_limit_is_part_of_ranking() -> None:
    def frequent(values: list[str], count: int) -> list[str]:
        counts: Any = Counter(values)
        return heapq.nlargest(count, counts, key=counts.get)

    program = trace(frequent, {"values": list_of(STR), "count": INT})
    assert isinstance(program.value, Call)
    assert program.value.op.name == "frequency_order"
    assert program.value.arguments[1] == Col("count", INT)


@pytest.mark.parametrize("backend", backend_params())
def test_frequency_ranking(backend: Backend[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def frequent(text: str, count: int) -> str:
        counts: Any = Counter(word.lower() for word in text.split())
        return " ".join(heapq.nlargest(count, counts, key=counts.get))

    inputs = []
    key = OpKey(".lower", (Kind.STR,))
    original = backend.rules.get
    rule = original(key)
    assert rule is not None

    def lower(value: Any, **facts: Any) -> Any:
        inputs.append(value)
        return rule(value, **facts)

    monkeypatch.setattr(
        backend.rules, "get", lambda candidate: lower if candidate == key else original(candidate)
    )
    rows = [
        (text, count)
        for text in (
            "",
            "B a b A c",
            "z a z b a b",
            "İ i 中文 中文 😀",
            "x\x00y x\x00y",
            "\u0085\u039fΣ\u2003\u039fς\u001cİ A\u0301Σ",
            " ".join([f"word{index}" for index in range(80, -1, -1)] * 2),
            " ".join(["winner"] * 8 + ["second", "third"] * 6 + ["last"]),
        )
        for count in (-(2**63), -3, 0, 1, 2, 3, 10, 2**63 - 1)
    ]
    assert check_function(frequent, {"text": STR, "count": INT}, backend, rows) == []
    if backend.name != "spark":
        assert inputs and all("text" in str(value) for value in inputs)


@pytest.mark.parametrize("backend", backend_params())
def test_frequency_source_is_bound_once(backend: Backend[Any]) -> None:
    if backend.name == "polars" or (backend.name == "datafusion" and not backend.supports_graphs()):
        pytest.skip("Frequency source binding requires lambda capture")
    source = Col("frequency_input", list_of(STR))
    with lowering_context(binds=True):
        lowered = backend.lower(apply("frequency_order", source, lit(5)))
    assert str(lowered.expr).count("frequency_input") == 1
    assert backend.evaluate([lowered.expr], {"frequency_input": list_of(STR)}, [([],)]) == [[[]]]
