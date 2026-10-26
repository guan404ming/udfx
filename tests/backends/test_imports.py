import builtins
import sys
from functools import partial
from types import FunctionType, ModuleType
from typing import Any

import pytest
from conftest import backend_params
from pyspark import cloudpickle

from udfx import compile
from udfx.backends.base import Backend
from udfx.errors import Unsupported
from udfx.frontend import trace
from udfx.types import STR


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize(
    "statement",
    ["import udfx_import_example as settings", "from udfx_import_example import pattern"],
)
def test_frozen_local_imports(
    backend: Backend[Any], helper: bool, statement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("udfx_import_example")
    vars(module)["pattern"] = "a"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace: dict[str, Any] = {}
    pattern = "settings.pattern" if statement.startswith("import") else "pattern"
    exec(
        f"def substitute(text):\n {statement}\n import re\n"
        f" return re.sub({pattern}, '#', text, count=2)\n"
        "def outer(text):\n return substitute(text)",
        namespace,
    )
    function = namespace["outer" if helper else "substitute"]
    translation = compile(function, {"text": STR}, backend)
    assert translation.program is not None, translation.unsupported
    vars(module)["pattern"] = "b"
    rows = [("aabb",), ("ab",)]
    assert translation.program.run(rows) == ["##bb", "#b"]
    assert cloudpickle.loads(cloudpickle.dumps(translation.program)).run(rows) == ["##bb", "#b"]
    assert backend.evaluate([translation.expr], {"text": STR}, rows) == [["##bb", "#b"]]


@pytest.mark.parametrize("statement", ["import udfx_module_not_loaded", "from . import sibling"])
def test_import_does_not_load_modules(statement: str) -> None:
    namespace: dict[str, Any] = {}
    exec(f"def rejected(text):\n {statement}\n return text", namespace)
    with pytest.raises(Unsupported, match="import"):
        trace(namespace["rejected"], {"text": STR})


def test_custom_importer_is_not_executed() -> None:
    namespace: dict[str, Any] = {}
    exec("def rejected(text):\n import re\n return text", namespace)
    calls = []
    custom = {**vars(builtins), "__import__": lambda *args: calls.append(args)}
    function = FunctionType(namespace["rejected"].__code__, {"__builtins__": custom})
    with pytest.raises(Unsupported, match="standard absolute"):
        trace(function, {"text": STR})
    assert calls == []


def test_replaced_builtin_importer_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace: dict[str, Any] = {}
    exec("def rejected(text):\n import re\n return text", namespace)
    original = builtins.__import__
    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", partial(original))
        with pytest.raises(Unsupported, match="standard absolute"):
            trace(namespace["rejected"], {"text": STR})
