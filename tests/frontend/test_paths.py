# ruff: noqa: PLW0108
import os
from collections.abc import Callable
from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from udfx.backends.polars import PolarsBackend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.ir import FALSE, Col
from udfx.testing import check_function
from udfx.types import BOOL, FLOAT, INT, STR, Type

PATHS = (
    "",
    "name",
    "/b",
    "/",
    "//",
    "///",
    "a/",
    "a//b",
    "a//",
    "a/b/",
    ".hidden",
    "..",
    "...",
    "...jpg",
    "a.b/c",
    "a.b/c.txt",
    "a/.hidden",
    "a/...hidden.txt",
    "a/b.",
    "a\n/b.jpg",
    "a/😀.jpg",
    "a/\x00.jpg",
)


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize(
    "function",
    [
        lambda text: os.path.join("a", text),
        lambda text: os.path.join(text, "train", text, f"{text}.jpg"),
        lambda text: os.path.join(text),
        lambda text: os.path.join(text, ""),
        lambda text: os.path.basename(text),
        lambda text: os.path.dirname(text),
        lambda text: os.path.splitext(text)[0],
        lambda text: os.path.splitext(text)[1],
        lambda text: os.path.split(text)[0],
        lambda text: os.path.split(text)[1],
    ],
)
def test_path_traps(function: Callable[[str], str], backend: Backend[Any]) -> None:
    assert (
        check_function(
            function, {"text": STR.or_none()}, backend, [(path,) for path in (*PATHS, None)]
        )
        == []
    )


def test_path_parts_unpack() -> None:
    def function(text: str) -> str:
        directory, name = os.path.split(text)
        root, extension = os.path.splitext(name)
        return directory + "|" + root + "|" + extension

    assert (
        check_function(function, {"text": STR}, PolarsBackend(), [(path,) for path in PATHS]) == []
    )


def test_fspath_is_identity() -> None:
    program = trace(lambda text: os.fspath(text), {"text": STR})
    assert program.value == Col("text", STR)
    assert program.fault == FALSE


@pytest.mark.parametrize(
    ("type", "value"), [(INT, 1), (FLOAT, 1.5), (BOOL, False), (STR.or_none(), None)]
)
def test_fspath_type_error(type: Type, value: object) -> None:
    def function(text: Any) -> str:
        try:
            return os.fspath(text)
        except TypeError:
            return "invalid"

    assert check_function(function, {"text": type}, PolarsBackend(), [(value,)]) == []


@pytest.mark.parametrize(
    "function", [os.stat, os.lstat, os.listdir, os.scandir, os.remove, open, os.path.isfile]
)
def test_file_system_access_stays_unsupported(function: Callable[..., object]) -> None:
    with pytest.raises(Unsupported, match="file system access"):
        trace(lambda text: function(text), {"text": STR})
