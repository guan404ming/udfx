import importlib
import subprocess
import sys
from importlib.metadata import version
from types import ModuleType

import pytest
from hypothesis import given

import udfx
from udfx import compile
from udfx.backends import installed, named
from udfx.backends.polars import PolarsBackend
from udfx.testing import batches, check_function
from udfx.types import INT, STR

POLARS = PolarsBackend()

IMPORT = importlib.import_module


def test_unsupported_is_reported() -> None:
    translation = compile(lambda s: s.expandtabs(), {"s": STR})
    assert not translation.ok
    assert translation.unsupported is not None


def test_schema_values_must_be_types() -> None:
    translation = compile(lambda a: a.lower(), {"a": str})  # pyright: ignore[reportArgumentType]
    assert str(translation.unsupported) == (
        "schema value for 'a' must be a udfx.types.Type such as STR or INT.or_none()"
    )


def test_public_surface() -> None:
    assert sorted(udfx.__all__) == sorted(
        ["Translation", "Unsupported", "__version__", "compile", "explain", "types"]
    )
    assert udfx.__version__ == version("udfx")
    assert udfx.types.STR == STR


def test_translation_repr() -> None:
    assert repr(compile(lambda s: s.upper(), {"s": STR})) == (
        "Translation(ok=True, backend='polars', result=str)"
    )
    assert repr(compile(lambda s: s.expandtabs(), {"s": STR})) == (
        "Translation(ok=False, backend='polars', unsupported='no op .expandtabs(str)')"
    )


def test_unknown_backend() -> None:
    message = "^unknown backend 'nope'; known backends: datafusion, duckdb, polars, spark$"
    with pytest.raises(ValueError, match=message):
        named("nope")


def test_missing_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named 'polars'")

    monkeypatch.setattr(importlib, "import_module", missing)
    message = r"^backend 'polars' needs its engine: pip install 'udfx\[polars\]'$"
    with pytest.raises(ImportError, match=message):
        compile(lambda s: s.lower(), {"s": STR})


def test_installed_skips_broken_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> ModuleType:
        if name == "udfx.backends.spark":
            raise RuntimeError("no Java")
        return IMPORT(name)

    monkeypatch.setattr(importlib, "import_module", broken)
    assert "spark" not in [backend.name for backend in installed()]


def test_installed_starts_no_spark_session() -> None:
    script = (
        "from pyspark import SparkContext\n"
        "from udfx.backends import installed\n"
        "installed()\n"
        "assert SparkContext._active_spark_context is None\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@given(batches({"s": STR.or_none()}))
def test_nullable_lower_matches_python(rows: list[tuple[object, ...]]) -> None:
    assert check_function(lambda s: s.lower(), {"s": STR.or_none()}, POLARS, rows) == []


@given(batches({"a": INT, "b": INT}))
def test_overflow_falls_back(rows: list[tuple[object, ...]]) -> None:
    assert check_function(lambda a, b: a + b > 0, {"a": INT, "b": INT}, POLARS, rows) == []
