from collections.abc import Callable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.core.values import MAX_UNROLL
from udfx.testing import check_function
from udfx.types import FLOAT, STR


def range_loop(text: str) -> str:
    for index in range(3):
        text += str(index)
    return text


def zip_loop(text: str) -> str:
    for prefix, number in zip(("a", "b"), (1, 2, 3), strict=False):
        text += prefix + str(number)
    return text


def enumerate_loop(text: str) -> str:
    for index, word in enumerate([text, "b"], 2):
        text += word + str(index)
    return text


def dict_loop(text: str) -> str:
    for key, value in {"a": 1, "b": 2}.items():
        text += key + str(value)
    return text


@pytest.mark.parametrize(
    "function",
    [
        range_loop,
        zip_loop,
        enumerate_loop,
        dict_loop,
        lambda text: [text + str(index) for index in range(3)],
        lambda text: [text + str(index) for index in range(1, 4)],
        lambda text: [text + str(index) for index in range(4, -1, -2)],
        lambda text: [text for _index in range(-1)],
        lambda text: [text + str(index) for index, _word in enumerate(("a", "b"), -1)],
        lambda text: [
            text + word + str(number) for word, number in zip(("a", "b"), (1,), strict=False)
        ],
        lambda text: [text + key + str(value) for key, value in {"a": 1, "b": 2}.items()],
        lambda text: [text + key for key in {"a": 1, "b": 2}.keys()],  # noqa: SIM118
        lambda text: [text + str(value) for value in {"a": 1, "b": 2}.values()],
        lambda text: text in {"a": 1, "b": 2}.keys(),  # noqa: SIM118
        lambda text: len(text) in {"a": 1, "b": 2}.values(),
        lambda text: (text, 1) in {"a": 1, "b": 2}.items(),
        lambda text: [1, 2] in {1: 2}.items(),  # pyright: ignore[reportOperatorIssue]
        lambda text: text in {"a": 1}.items(),
        lambda text: [pair[0] + text for pair in {"a": 1}.items()],
        lambda text: dict(zip(("a", "b"), (1, 2), strict=False)).get(text, 0),
        lambda text: list(range(3)),
        lambda text: list(tuple(("a", "b"))),
        lambda text: dict(a=1, b=2).get(text, 0),
        lambda text: dict((("a", 1), ("b", 2))).get(text, 0),
    ],
)
def test_constant_iterables(function: Callable[[str], object]) -> None:
    assert check_function(function, {"text": STR}, PolarsBackend(), [("",), ("a",), ("bc",)]) == []


def test_range_unroll_limit() -> None:
    def oversized(text: str) -> list[str]:
        return [text for _index in range(MAX_UNROLL + 1)]

    with pytest.raises(Unsupported, match="unrolled items"):
        trace(oversized, {"text": STR})


def word_length(word: str) -> int:
    return len(word) if word else -1


@pytest.mark.parametrize(
    "function",
    [
        lambda text: [str(index) + word for index, word in enumerate(text.split(), start=2)],
        lambda text: [str(index) + word for index, word in enumerate(text.split(), len(text))],
        lambda text: [str(index) + word for index, word in enumerate(("a", "b"), len(text))],
        lambda text: [left + right for left, right in zip(text.split(), ("a", "b"), strict=False)],
        lambda text: [
            left + right for left, right in zip(text.split(), text.split(","), strict=False)
        ],
        lambda text: [index + 1 for index in range(len(text))],
        lambda text: [index + 1 for index in range(1, len(text), 2)],
        lambda text: [index for index in range(5, len(text), -1)],
        lambda text: list(map(lambda word: word.upper(), text.split())),
        lambda text: list(map(word_length, text.split(","))),
        lambda text: list(map(int, text.split())),
        lambda text: list(map(lambda _word: 1 // 0, text.split())),
        lambda text: list(filter(lambda _word: 1 // 0, text.split())),
        lambda text: list(map(lambda character: character.upper(), text)),
        lambda text: list(filter(lambda character: character != "a", text)),
        lambda text: list(filter(lambda word: len(word) > 1, text.split())),
        lambda text: list(filter(None, text.split(","))),
        lambda text: list(map(lambda character: {"a": 1, "b": 2}[character], list(text))),
        lambda text: [word.upper() for word in filter(lambda word: word != "a", text.split())],
        lambda text: list(filter(lambda word: word[0] == "a", text.split(","))),
        lambda text: list(map(lambda word: word[0], text.split(","))),
        lambda text: list(map(word_length, ("a", ""))),
        lambda text: list(map(word_length, ())),
        lambda text: len(set(text)),
        lambda text: len(set([len(text), 1, 1])),
        lambda text: len({len(text), 1}),
        lambda text: len(set([float(len(text)), 1.0, 1.0])),
        lambda text: text.split() * 3,
        lambda text: -2 * text.split(),
        lambda text: [text, "a"] * len(text),
        lambda text: [word.upper() for word in tuple(text.split())],
    ],
)
def test_dynamic_iterables(function: Callable[[str], object]) -> None:
    rows = [("",), ("a",), ("a bb c",), ("a,b,",), ("12",), ("😀a😀",)]
    assert check_function(function, {"text": STR}, PolarsBackend(), rows) == []


def test_dynamic_range_requires_comprehension() -> None:
    with pytest.raises(Unsupported, match="dynamic iterators require a comprehension"):
        trace(lambda text: list(range(len(text))), {"text": STR})


def test_stored_constant_range() -> None:
    def repeated(text: str) -> str:
        indices = range(3)
        for index in indices:
            text += str(index)
        return text

    assert check_function(repeated, {"text": STR}, PolarsBackend(), [("a",)]) == []


def test_stored_zip_stays_unsupported() -> None:
    def deferred(text: str) -> list[str]:
        pairs = zip(text.split(), ("a", "b"), strict=False)
        return [left + right for left, right in pairs]

    with pytest.raises(Unsupported, match="consumed immediately"):
        trace(deferred, {"text": STR})


@pytest.mark.parametrize(
    "function",
    [
        lambda text: [
            pair for pair, _word in zip(enumerate(text.split()), text.split(), strict=False)
        ],
        lambda text: text in (text, "a"),
    ],
)
def test_iterator_forms_do_not_change_results(function: Callable[[str], object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"text": STR})


def test_empty_dictionary_views() -> None:
    def membership(text: str) -> bool:
        return text in {}.keys() or len(text) in {}.values()  # noqa: SIM118

    assert check_function(membership, {"text": STR}, PolarsBackend(), [("",), ("a",)]) == []


def numbered_loop(text: str) -> str:
    result = "prefix:"
    for index, word in enumerate(text.split()):
        if word == "a":
            continue
        result += str(index) + word
    return result


def paired_loop(text: str) -> int:
    result = 3
    for word, number in zip(text.split(), (1, 2), strict=False):
        result += len(word) * number
    return result


@pytest.mark.parametrize("function", [numbered_loop, paired_loop])
def test_dynamic_paired_loops(function: Callable[[str], object]) -> None:
    rows = [("",), ("a",), ("a bb c",), ("one two",)]
    assert check_function(function, {"text": STR}, PolarsBackend(), rows) == []


def test_loop_carried_condition_stays_unsupported() -> None:
    def conditional(text: str) -> int:
        result = 0
        for index, word in enumerate(text.split()):
            if result < 2:
                result += index + len(word)
        return result

    with pytest.raises(Unsupported, match="depend on the accumulator"):
        trace(conditional, {"text": STR})


def test_dynamic_range_loop_stays_unsupported() -> None:
    def accumulated(text: str) -> int:
        result = 0
        for index, _value in enumerate(range(len(text))):
            result += index
        return result

    with pytest.raises(Unsupported, match="loops and comprehensions"):
        trace(accumulated, {"text": STR})


def test_constant_dictionary_items_preserve_nan_identity() -> None:
    missing = float("nan")

    def membership(text: str) -> bool:
        return (missing, 1) in {missing: 1}.items()

    assert check_function(membership, {"text": STR}, PolarsBackend(), [("a",)]) == []


def test_dynamic_dictionary_items_with_nan_stay_unsupported() -> None:
    missing = float("nan")

    def membership(value: float) -> bool:
        return (value, 1) in {missing: 1}.items()

    with pytest.raises(Unsupported, match="non-reflexive"):
        trace(membership, {"value": FLOAT})
