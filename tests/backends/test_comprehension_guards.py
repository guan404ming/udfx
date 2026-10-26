from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.frontend import trace
from udfx.testing import check_function
from udfx.types import STR, list_of


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_catches_element_fault(backend: Backend[Any]) -> None:
    def mapped(text: str) -> list[str]:
        try:
            return [word.split("=")[1] for word in text.split(";") if word]
        except IndexError:
            return ["missing"]

    rows = [("a=b;c=d",), ("a=b;missing",), ("missing;a=b",), ("",)]
    assert check_function(mapped, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_failed_comprehension_restores_the_local(backend: Backend[Any]) -> None:
    def mapped(text: str) -> str:
        word = "saved"
        try:
            values = [word[0] for word in text.split(",")]
        except IndexError:
            return word
        return word + "".join(values)

    rows = [("a,b",), ("a,",), (",a",), ("",)]
    assert check_function(mapped, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_nested_comprehension_keeps_outer_capture(backend: Backend[Any]) -> None:
    def mapped(text: str) -> list[str]:
        def collect(word: str) -> str:
            return "".join([part + word for part in word.split(",")])

        return [collect(word) for word in text.split(";")]

    rows = [("a,b;c,d",), ("",), ("one;two",)]
    assert check_function(mapped, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_nonempty_comprehension_index(backend: Backend[Any]) -> None:
    def initials(text: str) -> str:
        try:
            return "".join([word[0] for word in text.split(" ") if word != ""])
        except IndexError:
            return "missing"

    rows = [("",), (" a  bee ",), ("é 中文",), ("\x00 x",)]
    assert check_function(initials, {"text": STR}, backend, rows) == []
    assert backend.lower_program(trace(initials, {"text": STR})).exact is None


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_keeps_helper_scope(backend: Backend[Any]) -> None:
    def mapped(text: str) -> list[str]:
        suffix = "!"

        def append(word: str) -> str:
            return word + suffix

        def cleaned(word: str) -> str:
            return append(word.lower())

        return [cleaned(word) for word in text.split()]

    rows = [("",), ("A Bee",), ("é 中文",)]
    assert check_function(mapped, {"text": STR}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("equal", [False, True])
def test_empty_comparison_keeps_null(backend: Backend[Any], equal: bool) -> None:
    def compared(text: str | None) -> bool:
        return text == "" if equal else text != ""

    rows = [(None,), ("",), ("a",)]
    assert check_function(compared, {"text": STR.or_none()}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_catches_first_element_exception(backend: Backend[Any]) -> None:
    def mapped(values: list[str | None]) -> list[int]:
        try:
            return [int(value.split("=")[1]) for value in values]  # pyright: ignore[reportOptionalMemberAccess]
        except IndexError:
            return [-1]
        except ValueError:
            return [-2]
        except AttributeError:
            return [-3]

    rows = [
        ([],),
        (["a=1", "b=2"],),
        (["missing", "a=invalid", None],),
        (["a=invalid", "missing", None],),
        ([None, "missing", "a=invalid"],),
        (["a=1", "missing"],),
        (["a=1", "a=invalid"],),
        (["a=1", None],),
        (["a=٣", "b=4"],),
    ]
    schema = {"values": list_of(STR.or_none())}
    assert check_function(mapped, schema, backend, rows) == []
    lowered = backend.lower_program(trace(mapped, schema))
    flags = backend.evaluate([backend.exact_expression(lowered)], schema, rows)[0]
    assert any(flag is True for flag in flags[2:8])


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_filter_exception_precedes_value(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[int]:
        try:
            return [int(value) for value in values if value[0] != "#"]
        except IndexError:
            return [-1]
        except ValueError:
            return [-2]

    rows = [([],), (["#bad", "2"],), (["", "bad"],), (["bad", ""],), (["1", ""],)]
    assert check_function(mapped, {"values": list_of(STR)}, backend, rows) == []


@pytest.mark.parametrize("backend", backend_params())
def test_comprehension_keeps_uncaught_exception(backend: Backend[Any]) -> None:
    def mapped(values: list[str]) -> list[int]:
        try:
            return [int(value.split("=")[1]) for value in values]
        except IndexError:
            return [-1]

    rows = [(["missing", "a=bad"],), (["a=bad", "missing"],), (["a=1"],)]
    assert check_function(mapped, {"values": list_of(STR)}, backend, rows) == []
