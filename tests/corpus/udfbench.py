from __future__ import annotations

import inspect
import os
from pathlib import Path

from udfx.types import FLOAT, INT, STR, Type

from .cases import Case

SCHEMAS = {
    "converttoeuro": (FLOAT.or_none(), STR.or_none()),
    "frequentterms": (STR.or_none(), INT),
    "addnoise": (INT.or_none(),),
    "log_10": (FLOAT.or_none(),),
}


def load() -> list[Case] | None:
    root = os.environ.get("UDFX_UDFBENCH")
    if not root:
        return None
    cases: list[Case] = []
    for path in sorted(Path(root, "engines/pyspark/udfs/scalar").glob("*.py")):
        namespace: dict[str, object] = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        function = namespace[path.stem]
        if not callable(function):
            continue
        names = list(inspect.signature(function).parameters)
        types = SCHEMAS.get(path.stem, (STR.or_none(),) * len(names))
        schema: dict[str, Type] = dict(zip(names, types, strict=True))
        cases.append(Case(path.stem, function, schema, "fallback"))
    return cases
