import re
import unicodedata
from collections.abc import Callable
from contextlib import nullcontext

import pytest

from udfx import compile
from udfx.types import INT, STR, Type

NOTHING = nullcontext()

counter = 0


def for_loop(s: str) -> str:
    for character in s:
        return character
    return s


def comprehension(s: str) -> int:
    return len([character for character in s])


def while_loop(a: int) -> int:
    while a > 0:
        a = a - 1
    return a


def global_assignment(a: int) -> int:
    global counter  # noqa: PLW0603
    counter = a
    return a


def global_deletion(a: int) -> int:
    global counter  # noqa: PLW0603
    del counter
    return a


def local_import(a: int) -> int:
    import math  # noqa: PLC0415

    return math.floor(a)


def with_block(s: str) -> str:
    with NOTHING:
        return s


def raise_statement(s: str) -> str:
    raise ValueError


def generator(s: str):
    yield s


def attribute(a: int) -> int:
    return a.real


def mixed_returns(a: int) -> int | str:
    if a > 0:
        return 1
    return "x"


def module_callable(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def unpacking(s: str) -> str:
    first, second = s
    return first + second


@pytest.mark.parametrize(
    ("function", "schema", "message"),
    [
        (for_loop, {"s": STR}, r"loops and comprehensions at for_loop\+\d+"),
        (comprehension, {"s": STR}, r"loops and comprehensions at comprehension\+\d+"),
        (while_loop, {"a": INT}, r"loops and comprehensions at while_loop\+\d+"),
        (global_assignment, {"a": INT}, r"global assignment at global_assignment\+\d+"),
        (global_deletion, {"a": INT}, r"global assignment at global_deletion\+\d+"),
        (local_import, {"a": INT}, r"imports at local_import\+\d+"),
        (with_block, {"s": STR}, r"with blocks at with_block\+\d+"),
        (raise_statement, {"s": STR}, r"raise statements at raise_statement\+\d+"),
        (generator, {"s": STR}, r"generators at generator\+\d+"),
        (
            attribute,
            {"a": INT},
            r"attribute access on a column value \(attribute load real\) at attribute\+\d+",
        ),
        (
            mixed_returns,
            {"a": INT},
            r"return types differ: returns int on one path and str on another; return one type",
        ),
        (module_callable, {"s": STR}, r"unknown callable unicodedata\.normalize"),
        (unpacking, {"s": STR}, r"opcode UNPACK_SEQUENCE at unpacking\+\d+"),
        (len, {"s": STR}, r"len is not a Python function"),
    ],
)
def test_message_names_the_feature(
    function: Callable[..., object], schema: dict[str, Type], message: str
) -> None:
    translation = compile(function, schema)
    assert re.fullmatch(message, str(translation.unsupported))
