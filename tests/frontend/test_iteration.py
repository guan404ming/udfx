from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.values import MAX_UNROLL
from udfx.testing import check_function
from udfx.types import STR


def constant_loop(text: str) -> str:
    result = text
    for word in ("a", "b"):
        result += word
    return result


def list_loop(text: str) -> str:
    result = ""
    for word in ["a", "b"]:
        if word in text:
            continue
        result += word
        if result == text:
            break
    else:
        result += "!"
    return result


def empty_loop(text: str) -> str:
    for word in ():
        text += word
    return text


@pytest.mark.parametrize("function", [constant_loop, list_loop, empty_loop])
def test_constant_loops(function: Callable[[str], str]) -> None:
    rows = [("",), ("a",), ("b",), ("ab",), ("x",)]
    assert check_function(function, {"text": STR}, PolarsBackend(), rows) == []


def test_loop_limit() -> None:
    words = ("a",) * (MAX_UNROLL + 1)

    def oversized(text: str) -> str:
        for word in words:
            text += word
        return text

    with pytest.raises(Unsupported, match="unrolled items"):
        trace(oversized, {"text": STR})


def test_dynamic_loop_stays_unsupported() -> None:
    def dynamic(text: str) -> str:
        result = ""
        for word in text.split():
            result += word
        return result

    with pytest.raises(Unsupported, match="loops and comprehensions"):
        trace(dynamic, {"text": STR})


@pytest.mark.parametrize(
    "function",
    [
        lambda text: [word.upper() for word in text.split()],
        lambda text: [word.split("=")[1] for word in text.split(";") if word.startswith("Func")],
        lambda text: " ".join([word for word in text.split() if word not in {"a", "the"}]),
        lambda text: [word for word in text.split()[1:] if word],
        lambda text: [word.upper() if len(word) > 1 else word for word in text.split()],
        lambda text: any(word in text for word in ("a", "b")),
        lambda text: any(word[0] in text for word in ("a", "")),
        lambda text: all(word[0] != "a" for word in text.split(",")),
        lambda text: sum(len(word) for word in text.split()),
        lambda text: sum(1 for word in text.split() if word != "a"),
        lambda text: [1 for _word in text.split()],
        lambda text: "-".join("x" for _word in text.split()),
        lambda text: sum(int(word) for word in text.split()),
        lambda text: len([word for word in text.split() if word != "a"]),
        lambda text: len(set(text.split())),
        lambda text: sorted(set(text.split())),
        lambda text: list(set(text.split())),
        lambda text: sorted(text.split()),
        lambda text: list(text.split()),
        lambda text: list(word.upper() for word in text.split()),
        lambda text: "-".join(word.upper() for word in text.split()),
        lambda text: [text, text.upper()],
        lambda text: [len(text), 1],
        lambda text: [1, 2, 3],
        lambda text: len(text) in [1, 2, 3],
        lambda text: float(len(text)) in [1.0, 2.0, 3.0],
        lambda text: bool(text) in [True, False],
        lambda text: [word + text for word in ("a", "b")],
        lambda text: any(text.split()),
        lambda text: all(text.split()),
        lambda text: [],
        lambda text: [word for word in text.split() if False],
        lambda text: any(word[0] for word in text.split() if False),
        lambda text: [text.split(), text.split(";")],
        lambda text: len({text, "a"}),
        lambda text: text in {text.upper(), "a"},
        lambda text: sorted({text, "a"}),
        lambda text: bool(set(text.split())),
        lambda text: ",".join(set(text.split())),
    ],
)
def test_list_expressions(function: Callable[[str], object]) -> None:
    rows = [("",), ("a",), ("b c a",), ("Func=a;Other=b;Func=c",), ("Func",), ("1 2",), ("a,",)]
    assert check_function(function, {"text": STR}, PolarsBackend(), rows) == []


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda text: [text, len(text)], "list items must share one type"),
        (
            lambda text: [word for part in text.split() for word in part.split()],
            "loops and comprehensions",
        ),
        (lambda text: (word for word in text.split()), "expected a value"),
        (lambda text: set(text.split()), "expected a value"),
        (lambda text: len(word for word in text.split()), "expected a value"),  # pyright: ignore[reportArgumentType]
    ],
)
def test_unsupported_list_forms(function: Callable[[str], object], message: str) -> None:
    with pytest.raises(Unsupported, match=message):
        trace(function, {"text": STR})


def test_stored_generator_stays_unsupported() -> None:
    def deferred(text: str) -> bool:
        words = (word in text for word in ("a", "b"))
        text = "changed"
        return any(words)

    with pytest.raises(Unsupported, match="consumed immediately"):
        trace(deferred, {"text": STR})


def test_while_loop_stays_unsupported() -> None:
    def repeated(text: str) -> str:
        while text:
            text = text[1:]
        return text

    with pytest.raises(Unsupported, match="loops and comprehensions"):
        trace(repeated, {"text": STR})


def test_comprehension_restores_local() -> None:
    def restored(text: str) -> str:
        word = "saved"
        values = [word.upper() for word in text.split()]
        return word + ",".join(values)

    assert check_function(restored, {"text": STR}, PolarsBackend(), [("",), ("a b",)]) == []


def test_comprehension_fault_handlers_stay_unsupported() -> None:
    def guarded(text: str) -> list[str]:
        try:
            return [word.split("=")[1] for word in text.split(";")]
        except IndexError:
            return ["missing"]

    with pytest.raises(Unsupported, match="comprehension faults with exception handlers"):
        trace(guarded, {"text": STR})


def test_constant_loop_accepts_limit() -> None:
    words = ("a",) * MAX_UNROLL

    def repeated(text: str) -> str:
        for word in words:
            text += word
        return text

    assert check_function(repeated, {"text": STR}, PolarsBackend(), [("",), ("b",)]) == []


def test_inplace_list_mutation_stays_unsupported() -> None:
    def mutate(text: str) -> list[str]:
        values = [text]
        alias = values
        values += ["added"]
        return alias

    with pytest.raises(Unsupported, match="in-place list mutation"):
        trace(mutate, {"text": STR})


def test_literal_reduction_limit() -> None:
    words = ("a",) * (MAX_UNROLL + 1)

    def oversized(text: str) -> bool:
        return any(words)

    with pytest.raises(Unsupported, match="unrolled items"):
        trace(oversized, {"text": STR})


def test_large_integer_in_comprehension_uses_fallback() -> None:
    numbers = (2**80,)

    def spelling(text: str) -> list[str]:
        return [str(number) + text for number in numbers]

    assert check_function(spelling, {"text": STR}, PolarsBackend(), [("",)]) == []


def test_generator_fault_handler_cannot_change_result_type() -> None:
    def guarded(text: str) -> int | str:
        try:
            return sum(int(word) for word in text.split())
        except ValueError:
            return "invalid"

    with pytest.raises(Unsupported, match="comprehension faults with exception handlers"):
        trace(guarded, {"text": STR})
