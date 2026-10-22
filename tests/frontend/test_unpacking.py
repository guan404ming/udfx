import dis
from collections.abc import Callable
from types import ModuleType

import pytest

from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.frontend.opcodes import py312, py313, py314
from udfx.ir import FALSE, Col, Lit, lit
from udfx.ops import apply
from udfx.testing import check_function
from udfx.types import INT, STR, list_of

POLARS = PolarsBackend()


def unpack_pair(items: list[str]) -> str:
    first, second = items
    return first + second


def unpack_split(text: str) -> str:
    first, second = text.split(",")
    return first + second


def catch_unpack(items: list[str]) -> str:
    first = "unchanged"
    try:
        first, second = items
        return first + second
    except ValueError:
        return first
    except TypeError:
        return "missing"
    except IndexError:
        return "wrong fault"


def test_unpack_sequence_uses_one_length_guard() -> None:
    source = Col("items", list_of(STR))
    program = trace(unpack_pair, {"items": source.type})
    assert program.value == apply(
        "__add__", apply("__getitem__", source, lit(0)), apply("__getitem__", source, lit(1))
    )
    assert program.fault == apply("__ne__", apply("len", source), lit(2))


@pytest.mark.parametrize("function", [unpack_pair, catch_unpack])
def test_unpack_list_faults(function: Callable[..., object]) -> None:
    rows = [(None,), ([],), (["a"],), (["a", "b"],), (["a", "b", "c"],)]
    assert check_function(function, {"items": list_of(STR).or_none()}, POLARS, rows) == []


def test_unpack_split_faults() -> None:
    rows = [(None,), ("",), ("a",), ("a,b",), ("a,b,c",)]
    assert check_function(unpack_split, {"text": STR.or_none()}, POLARS, rows) == []


@pytest.mark.parametrize("pair", [("a", "b"), (1, "b"), (None, "b"), ((1, 2), "b")])
def test_unpack_constant_tuple(pair: tuple[object, str]) -> None:
    def unpack(text: str) -> str:
        _first, second = pair
        return text + second

    program = trace(unpack, {"text": STR})
    assert program.value == apply("__add__", Col("text", STR), lit("b"))
    assert program.fault == FALSE


def test_unpack_frontend_tuple() -> None:
    def unpack(text: str) -> str:
        pair = (text, 2)
        first, second = pair
        return first[:second]

    program = trace(unpack, {"text": STR})
    assert program.value == apply("slice", Col("text", STR), Lit(None, INT.or_none()), lit(2))
    assert program.fault == FALSE


@pytest.mark.parametrize(
    ("count", "dynamic"), [(0, False), (1, False), (3, False), (1, True), (3, True)]
)
def test_unpack_tuple_wrong_length(count: int, dynamic: bool) -> None:
    pair = tuple("a" for _ in range(count))

    def unpack(text: str) -> str:
        values = ((text,) if count == 1 else (text, text, text)) if dynamic else pair
        try:
            first, second = values  # pyright: ignore[reportAssignmentType]
            return first + second
        except ValueError:
            return "wrong length"

    program = trace(unpack, {"text": STR})
    assert program.value == lit("wrong length")
    assert program.fault == FALSE


@pytest.mark.parametrize("value", ["ab", {"a", "b"}, {"a": 1, "b": 2}, 1, None])
@pytest.mark.parametrize("starred", [False, True])
def test_unpack_unsupported_source(value: object, starred: bool) -> None:
    def unpack(text: str) -> str:
        if starred:
            first, *_rest = value  # pyright: ignore[reportGeneralTypeIssues]
        else:
            first, _second = value  # pyright: ignore[reportGeneralTypeIssues]
        return text + first

    with pytest.raises(Unsupported, match="unpacking requires a list or tuple"):
        trace(unpack, {"text": STR})


@pytest.mark.parametrize("value", [(), (1, "a"), (None, None), ("a", "b")])
def test_constant_tuple_return_is_unsupported(value: tuple[object, ...]) -> None:
    with pytest.raises(Unsupported, match="tuple returns are unsupported; return a scalar or list"):
        trace(lambda text: value, {"text": STR})


def test_frontend_tuple_return_is_unsupported() -> None:
    with pytest.raises(Unsupported, match="tuple returns are unsupported; return a scalar or list"):
        trace(lambda text: (text, 1), {"text": STR})


@pytest.mark.parametrize("normalizer", [py312, py313, py314], ids=["3.12", "3.13", "3.14"])
@pytest.mark.parametrize("starred", [False, True])
def test_unpack_opcode_versions(
    monkeypatch: pytest.MonkeyPatch, normalizer: ModuleType, starred: bool
) -> None:
    def function(items: list[str]) -> list[str]:
        return items

    resume, load, returned = dis.get_instructions(function)
    load = load._replace(opname="LOAD_FAST_BORROW" if normalizer is py314 else "LOAD_FAST")
    unpack = load._replace(
        opname="UNPACK_EX" if starred else "UNPACK_SEQUENCE",
        arg=257 if starred else 2,
        argval=257 if starred else 2,
    )
    instructions = [resume, load]
    if starred:
        instructions.append(load._replace(opname="EXTENDED_ARG", arg=1, argval=1))
    instructions.append(unpack)
    if normalizer is py312:
        instructions.extend(
            load._replace(opname="STORE_FAST", argval=name) for name in ("first", "second")
        )
    else:
        instructions.append(
            load._replace(opname="STORE_FAST_STORE_FAST", arg=18, argval=("first", "second"))
        )
    if starred:
        instructions.append(load._replace(opname="STORE_FAST", argval="last"))
    instructions.extend([load._replace(argval="second"), returned])
    instructions = [part._replace(offset=2 * index) for index, part in enumerate(instructions)]
    monkeypatch.setattr(dis, "get_instructions", lambda code: iter(instructions))
    monkeypatch.setattr("udfx.frontend.context.normalize", normalizer.normalize)
    source = Col("items", list_of(STR))
    program = trace(function, {"items": source.type})
    assert program.value == (
        apply("slice", source, lit(1), lit(-1, INT.or_none()))
        if starred
        else apply("__getitem__", source, lit(1))
    )
