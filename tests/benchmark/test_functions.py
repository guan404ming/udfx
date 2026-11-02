import inspect
import math
import traceback
from pathlib import Path

import pytest

from workloads.functions import load_function


def write_source(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "source.py"
    path.write_text(source)
    return path


def test_loader_keeps_defaults_annotations_helpers_and_source(tmp_path: Path) -> None:
    source = """from math import floor as down
OFFSET: int = 3
UNUSED = missing_query()

def helper(value):
    return down(value) + OFFSET

def selected(value: float, /, offset: int = OFFSET, *, suffix: str = "!") -> str:
    return str(helper(value) + offset) + suffix

def unused_query():
    raise RuntimeError("query executed")

start_engine()
"""
    path = write_source(tmp_path, source)
    function = load_function(path, "selected")
    assert function(2.8) == "8!"
    assert function(2.8, 1, suffix="?") == "6?"
    assert function.__annotations__ == {"value": float, "offset": int, "suffix": str, "return": str}
    assert (
        str(inspect.signature(function))
        == "(value: float, /, offset: int = 3, *, suffix: str = '!') -> str"
    )
    assert function.__code__.co_filename == str(path)
    assert function.__code__.co_firstlineno == 8
    assert path.read_text() == source


@pytest.mark.parametrize(
    "imports, decorator",
    [
        ("from pyspark.sql.functions import udf", "udf(StringType())"),
        ("from pyspark.sql.functions import udf as wrapper", "wrapper(returnType=StringType())"),
        ("import pyspark.sql.functions as functions", "functions.udf(StringType())"),
        ("from pyspark.sql.functions import udf", "udf"),
    ],
)
def test_loader_removes_only_recognized_spark_wrapper(
    tmp_path: Path, imports: str, decorator: str
) -> None:
    source = (
        f"{imports}\nfrom pyspark.sql.types import StringType\n"
        f"@{decorator}\ndef selected(value, suffix='?'):\n    return value + suffix\n"
    )
    function = load_function(write_source(tmp_path, source), "selected")
    assert function("hello") == "hello?"
    assert function("hello", "!") == "hello!"
    assert not {"udf", "wrapper", "functions", "StringType"} & function.__globals__.keys()


def test_loader_preserves_future_annotations_and_lexical_scopes(tmp_path: Path) -> None:
    source = """from __future__ import annotations
BASE = 4

def helper(value):
    return value + BASE

def selected(values: MissingType, offset=BASE):
    def nested(value):
        return helper(value) + offset
    return [nested(value) for value in values]
"""
    function = load_function(write_source(tmp_path, source), "selected")
    assert function([1, 2]) == [9, 10]
    assert function.__annotations__ == {"values": "MissingType"}


def test_loader_preserves_shared_literal_defaults_and_recursive_helpers(tmp_path: Path) -> None:
    source = """VALUES = [2]

def helper(value):
    return 0 if value == 0 else 1 + helper(value - 1)

def selected(value, values=VALUES):
    return helper(value) + values[0]
"""
    function = load_function(write_source(tmp_path, source), "selected")
    assert function(3) == 5
    assert function.__defaults__ is not None
    assert function.__defaults__[0] is function.__globals__["VALUES"]


def test_loader_keeps_real_traceback_filename_and_line(tmp_path: Path) -> None:
    path = write_source(tmp_path, "UNUSED = 2\n\ndef selected(value):\n    return 1 / value\n")
    function = load_function(path, "selected")
    with pytest.raises(ZeroDivisionError) as failure:
        function(0)
    frame = traceback.extract_tb(failure.value.__traceback__)[-1]
    assert (frame.filename, frame.lineno) == (str(path), 4)


@pytest.mark.parametrize(
    "source, reason",
    [
        ("def selected(value):\n    return sqrt(value)\n", "Missing source binding: sqrt"),
        ("VALUE = compute()\ndef selected():\n    return VALUE\n", "Nonliteral source binding"),
        ("VALUE: int\ndef selected():\n    return VALUE\n", "Missing source value"),
        (
            "VALUE = 1\ndef selected():\n    return globals()['VALUE']\n",
            "Dynamic source namespace",
        ),
        ("@unknown\ndef selected(value):\n    return value\n", "Unsupported decorator"),
        (
            "from pyspark.sql.functions import udf\n@udf\n@udf\n"
            "def selected(value):\n    return value\n",
            "Unsupported decorator",
        ),
        (
            "from pyspark.sql.functions import udf\n@udf(start_engine())\n"
            "def selected(value):\n    return value\n",
            "Unsupported decorator",
        ),
        (
            "from pyspark.sql.functions import udf\n@udf\ndef helper(value):\n"
            "    return value\ndef selected(value):\n    return helper(value)\n",
            "Decorated dependency",
        ),
        (
            "from pyspark.sql.functions import udf\n@udf\ndef selected(value):\n"
            "    return selected(value - 1) if value else 0\n",
            "Decorated dependency",
        ),
        ("def selected(value=start_engine()):\n    return value\n", "definition-time expression"),
        ("def selected(value: start_engine()):\n    return value\n", "definition-time expression"),
        ("import os\ndef selected():\n    return os.getcwd()\n", "Unsupported source import"),
        ("from math import *\ndef selected(value):\n    return abs(value)\n", "Wildcard"),
        ("VALUE = 1\nVALUE = 2\ndef selected():\n    return VALUE\n", "ambiguous source binding"),
        (
            "VALUES = [1]\nVALUES.append(2)\ndef selected():\n    return VALUES\n",
            "Top-level statement",
        ),
        ("if True:\n    VALUE = 1\ndef selected():\n    return VALUE\n", "source binding"),
        ("selected = lambda value: value\n", "Missing or ambiguous function"),
        ("async def selected(value):\n    return value\n", "Missing or ambiguous function"),
    ],
)
def test_loader_rejects_unresolved_or_effectful_definitions(
    tmp_path: Path, source: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        load_function(write_source(tmp_path, source), "selected")


def test_loader_accepts_duplicate_imports_without_importing_unused_modules(tmp_path: Path) -> None:
    source = """import math
import math
import missing_engine

def selected(value):
    return math.floor(value)
"""
    function = load_function(write_source(tmp_path, source), "selected")
    assert function(1.5) == 1
    assert "missing_engine" not in function.__globals__


def test_loader_requires_explicit_used_bindings_without_replacing_source(tmp_path: Path) -> None:
    source = (
        "OFFSET = 2\ndef selected(value, operation=ceil):\n    return operation(value) + OFFSET\n"
    )
    path = write_source(tmp_path, source)
    with pytest.raises(ValueError, match="Missing source binding: ceil"):
        load_function(path, "selected")
    function = load_function(path, "selected", bindings={"ceil": math.ceil})
    assert function(1.5) == 4
    assert function.__globals__["ceil"] is math.ceil
    assert function.__code__.co_filename == str(path)
    assert path.read_text() == source
    with pytest.raises(ValueError, match="must all be used"):
        load_function(path, "selected", bindings={"ceil": math.ceil, "floor": math.floor})
    for name in ("OFFSET", "len", "selected"):
        with pytest.raises(ValueError, match="must not replace"):
            load_function(path, "selected", bindings={"ceil": math.ceil, name: 0})
