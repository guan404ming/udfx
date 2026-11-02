from collections.abc import Callable

import pytest
from conftest import frontend_backend_params

from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR


def collect_words(text: str) -> list[str]:
    values = []
    for word in text.split(" "):
        if word and word != "skip":
            values.append(word.upper())
    return values


def collect_numbers(text: str) -> list[int]:
    values = []
    for word in text.split():
        values.append(int(word))
    return values


def concatenate_fragments(text: str) -> str:
    result = "prefix:"
    for word in text.split(" "):
        folded = word.lower()
        selected = folded == "replace"
        if selected:
            result += "* "
        else:
            result = result + word
            result = result + "! "
    return result.strip()


@pytest.mark.parametrize("backend", frontend_backend_params())
@pytest.mark.parametrize("function", [collect_words, collect_numbers, concatenate_fragments])
def test_local_collection_loops(function: Callable[..., object], backend: Backend) -> None:
    rows = [("",), (" ",), ("a skip b",), ("replace One 😀",), ("1 2 -3",), ("4 bad 5",)]
    assert check_function(function, {"text": STR}, backend, rows) == []


def aliased(text: str) -> list[str]:
    values = []
    alias = values
    for word in text.split():
        values.append(word)
    return alias


def nested_alias(text: str) -> int:
    values = []
    aliases = (values,)
    for word in text.split():
        values.append(word)
    return len(aliases[0])


def copied_initializer(text: str) -> int:
    aliases = ((values := []),)
    for word in text.split():
        values.append(word)
    return len(aliases[0])


def wrap_list(values: list[str]) -> tuple[list[str]]:
    return (values,)


def copied_helper_argument(text: str) -> int:
    aliases = wrap_list(values := [])
    for word in text.split():
        values.append(word)
    return len(aliases[0])


def append_argument(values: list[str], text: str) -> None:
    for word in text.split():
        values.append(word)


def mutated_helper_argument(text: str) -> int:
    values = []
    append_argument(values, text)
    return len(values)


def unused_old_alias(text: str) -> int:
    values = []
    aliases = (values,)
    values = []
    for word in text.split():
        values.append(word)
    return len(values) + len(aliases[0])


@pytest.mark.parametrize("backend", frontend_backend_params())
def test_fresh_list_replaces_an_escaped_list(backend: Backend) -> None:
    assert check_function(unused_old_alias, {"text": STR}, backend, [("a b",), ("",)]) == []


def dependent(text: str) -> list[str]:
    values = []
    for word in text.split():
        if len(values) < 2:
            values.append(word)
    return values


def repeated_append(text: str) -> list[str]:
    values = []
    for word in text.split():
        values.append(word)
        values.append(word.upper())
    return values


def escaped_target(text: str) -> str:
    word = "empty"
    values = []
    for word in text.split():
        values.append(word)
    return word


def escaped_temporary(text: str) -> str:
    result = ""
    for word in text.split():
        temporary = word.upper()
        result += temporary
    return temporary  # pyright: ignore[reportPossiblyUnboundVariable]


def carried_temporary(text: str) -> str:
    result = ""
    temporary = "initial"
    for word in text.split():
        result += temporary
        temporary = word
    return result


def partial_collection(text: str) -> int:
    values = []
    try:
        for word in text.split():
            values.append(int(word))
    except ValueError:
        return len(values)
    return len(values)


@pytest.mark.parametrize(
    "function",
    [
        aliased,
        nested_alias,
        copied_initializer,
        copied_helper_argument,
        mutated_helper_argument,
        dependent,
        repeated_append,
        escaped_target,
        escaped_temporary,
        carried_temporary,
        partial_collection,
    ],
)
def test_unproven_collection_loops_stay_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported):
        trace(function, {"text": STR})
