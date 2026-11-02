from collections.abc import Callable, Iterable

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


def joined(text: str) -> str:
    pieces = (word.lower() + "!" for word in text.split() if len(word) > 1)
    return "|".join(pieces)


def captured(text: str) -> str:
    suffix = text[:1]
    pieces = (word + suffix for word in text.split())
    return "|".join(pieces)


def listed(text: str) -> list[str]:
    pieces = (word.lower() for word in text.split())
    return list(pieces)


def ordered(text: str) -> list[str]:
    pieces = (word.lower() for word in text.split())
    return sorted(pieces)


def summed(text: str) -> int:
    lengths = (len(word) for word in text.split())
    return sum(lengths)


def mapped(text: str) -> list[str]:
    pieces = map(lambda word: word.lower(), text.split())
    return list(pieces)


def filtered(text: str) -> list[str]:
    pieces = filter(lambda word: len(word) > 1, text.split())
    return list(pieces)


def caught(text: str) -> str:
    try:
        pieces = (word.upper() for word in text.split())
        return "|".join(pieces)
    except IndexError:
        return "short"


def faulting(text: str) -> str:
    pieces = (word[1] for word in text.split(","))
    return "|".join(pieces)


def constant_fault(text: str) -> list[int]:
    numbers = (1 // 0 for _word in text.split())
    return list(numbers)


@pytest.mark.parametrize(
    "function",
    [joined, captured, listed, ordered, summed, mapped, filtered, caught, faulting, constant_fault],
)
def test_immediate_named_generator(function: Callable[[str], object]) -> None:
    rows = [("",), (" ",), ("A bb Cc",), ("İ Σς 😀",), ("a,bc,",)]
    assert check_function(function, {"text": STR}, PolarsBackend(), rows) == []


def late_capture(text: str) -> str:
    pieces = (word + text for word in ("a", "b"))
    text = "changed"
    return "|".join(pieces)


def reused(text: str) -> str:
    pieces = (word.lower() for word in text.split())
    result = "|".join(pieces)
    return result + "|".join(pieces)


def aliased(text: str) -> str:
    pieces = (word.lower() for word in text.split())
    alias = pieces
    return "|".join(alias)


def intervening_fault(text: str) -> str:
    pieces = (word[1] for word in text.split(","))
    int(text)
    return "|".join(pieces)


def conditional(text: str) -> str:
    pieces = (word[1] for word in text.split(","))
    if text:
        return "|".join(pieces)
    return "empty"


def different_handler(text: str) -> str:
    pieces = (word[1] for word in text.split(","))
    try:
        return "|".join(pieces)
    except IndexError:
        return "short"


def dynamic_separator(text: str) -> str:
    pieces = (word.lower() for word in text.split())
    return text.join(pieces)


def escaped(text: str) -> str:
    pieces = (word.lower() for word in text.split())

    def consume() -> str:
        return "|".join(pieces)

    return consume()


def unused_fault(text: str) -> str:
    pieces = (1 // 0 for _number in (1,))
    del pieces
    return text


def consume_twice(pieces: Iterable[str]) -> str:
    return "|".join(pieces) + "|".join(pieces)


def unknown_consumer(text: str) -> str:
    pieces = (word.lower() for word in text.split())
    return consume_twice(pieces)


@pytest.mark.parametrize(
    "function",
    [
        late_capture,
        reused,
        aliased,
        intervening_fault,
        conditional,
        different_handler,
        dynamic_separator,
        escaped,
        unused_fault,
        unknown_consumer,
    ],
)
def test_deferred_named_generator_rejected(function: Callable[[str], object]) -> None:
    with pytest.raises(Unsupported, match="consumed immediately"):
        trace(function, {"text": STR})


def copied(text: str) -> list[str]:
    aliases = ((pieces := (word.lower() for word in text.split())), list(pieces))
    return list(aliases[0])


def copied_across_block(text: str) -> list[str]:
    aliases = ((pieces := (word.lower() for word in text.split())), list(pieces), 1 if text else 0)
    return list(aliases[0])


@pytest.mark.parametrize("function", [copied, copied_across_block])
def test_copied_generator_rejected(function: Callable[[str], object]) -> None:
    with pytest.raises(Unsupported, match="consumed immediately"):
        trace(function, {"text": STR})
