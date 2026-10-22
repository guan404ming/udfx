from collections.abc import Callable

import pytest

from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import Col, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type, list_of

from .test_unpacking import POLARS


def unpack_rest(items: list[str]) -> list[str]:
    _first, *rest = items
    return rest


def unpack_prefix(items: list[str]) -> list[str]:
    *prefix, _last = items
    return prefix


def unpack_middle(items: list[str]) -> str:
    first, *middle, last = items
    return first + ",".join(middle) + last


def unpack_multiple_ends(items: list[str]) -> str:
    first, second, *middle, penultimate, last = items
    return first + second + ",".join(middle) + penultimate + last


def unpack_all(items: list[str]) -> list[str]:
    (*rest,) = items
    return rest


def catch_starred(items: list[str]) -> str:
    first = "unchanged"
    try:
        first, *middle, last = items
        return first + ",".join(middle) + last
    except ValueError:
        return first
    except TypeError:
        return "missing"
    except IndexError:
        return "wrong fault"


@pytest.mark.parametrize(
    "function",
    [unpack_rest, unpack_prefix, unpack_middle, unpack_multiple_ends, unpack_all, catch_starred],
)
def test_unpack_starred_list(function: Callable[..., object]) -> None:
    rows = [(None,), *((list("abcdef"[:count]),) for count in range(7))]
    assert check_function(function, {"items": list_of(STR).or_none()}, POLARS, rows) == []


@pytest.mark.parametrize(
    ("function", "start", "stop", "minimum"),
    [(unpack_rest, 1, None, 1), (unpack_prefix, 0, -1, 1), (unpack_all, 0, None, 0)],
)
def test_unpack_starred_uses_slice(
    function: Callable[..., object], start: int, stop: int | None, minimum: int
) -> None:
    source = Col("items", list_of(STR))
    program = trace(function, {"items": source.type})
    assert program.value == apply("slice", source, lit(start), lit(stop, INT.or_none()))
    assert program.fault == apply("__lt__", apply("len", source), lit(minimum))


@pytest.mark.parametrize("item", [STR.or_none(), INT, FLOAT, BOOL], ids=str)
def test_unpack_list_item_types(item: Type) -> None:
    program = trace(unpack_rest, {"items": list_of(item)})
    assert program.result == list_of(item)


@pytest.mark.parametrize("pair", [("a", "b"), (1, "b"), ()])
def test_starred_tuple_is_unsupported(pair: tuple[object, ...]) -> None:
    def unpack(text: str) -> str:
        _first, *rest = pair
        return text + str(rest)

    with pytest.raises(Unsupported, match="starred unpacking requires a list"):
        trace(unpack, {"text": STR})


def test_starred_frontend_tuple_is_unsupported() -> None:
    def unpack(text: str) -> str:
        pair = (text, 1)
        first, *_rest = pair
        return first

    with pytest.raises(Unsupported, match="starred unpacking requires a list"):
        trace(unpack, {"text": STR})
