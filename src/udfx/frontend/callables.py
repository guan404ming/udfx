from __future__ import annotations

import math
import os
import posixpath
import re
from datetime import datetime

from udfx.errors import Unsupported
from udfx.frontend.values import named

CALLABLES: tuple[tuple[object, str], ...] = (
    (abs, "abs"),
    (any, "any"),
    (all, "all"),
    (bool, "bool"),
    (float, "float"),
    (int, "int"),
    (len, "len"),
    (list, "list"),
    (set, "set"),
    (sorted, "sorted"),
    (sum, "sum"),
    (max, "max"),
    (min, "min"),
    (round, "round"),
    (str, "str"),
    (str.maketrans, "str.maketrans"),
    (os.fspath, "os.fspath"),
    (posixpath.join, "posixpath.join"),
    (posixpath.basename, "posixpath.basename"),
    (posixpath.dirname, "posixpath.dirname"),
    (posixpath.splitext, "posixpath.splitext"),
    (posixpath.split, "posixpath.split"),
    (math.ceil, "math.ceil"),
    (math.floor, "math.floor"),
    (math.isinf, "math.isinf"),
    (math.isnan, "math.isnan"),
    (math.log, "math.log"),
    (math.log2, "math.log2"),
    (math.exp, "math.exp"),
    (math.log10, "math.log10"),
    (math.sqrt, "math.sqrt"),
    (re.compile, "re.compile"),
    (re.sub, "re.sub"),
    (re.search, "re.search"),
    (re.match, "re.match"),
    (re.fullmatch, "re.fullmatch"),
    (re.findall, "re.findall"),
    (re.split, "re.split"),
    (re.Pattern.sub, "re.sub"),
    (re.Pattern.search, "re.search"),
    (re.Pattern.match, "re.match"),
    (re.Pattern.fullmatch, "re.fullmatch"),
    (re.Pattern.findall, "re.findall"),
    (re.Pattern.split, "re.split"),
)


CALLABLE_IDENTITIES = {
    ("builtins", "pandas", "isna"): "pandas.isna",
    ("builtins", "pandas", "notna"): "pandas.notna",
    ("builtins", "pandas.core.dtypes.missing", "isna"): "pandas.isna",
    ("builtins", "pandas.core.dtypes.missing", "notna"): "pandas.notna",
    ("numpy", "ufunc", "isnan"): "numpy.isnan",
    ("numpy", "ufunc", "log"): "numpy.log",
    ("numpy", "ufunc", "log1p"): "numpy.log1p",
    ("numpy", "ufunc", "log2"): "numpy.log2",
    ("numpy", "ufunc", "log10"): "numpy.log10",
    ("numpy", "ufunc", "expm1"): "numpy.expm1",
    ("numpy", "ufunc", "sqrt"): "numpy.sqrt",
    ("numpy", "ufunc", "exp"): "numpy.exp",
    ("numpy", "ufunc", "absolute"): "numpy.abs",
    ("numpy", "ufunc", "floor"): "numpy.floor",
    ("numpy", "ufunc", "ceil"): "numpy.ceil",
    ("numpy", "numpy", "round"): "numpy.round",
}


def known_op(callee: object) -> str | None:
    if (
        getattr(callee, "__self__", None) is datetime
        and getattr(callee, "__name__", "") == "strptime"
    ):
        return "datetime.strptime"
    if isinstance(getattr(callee, "__self__", None), re.Pattern):
        callee = getattr(re.Pattern, getattr(callee, "__name__", ""), callee)
    for known, name in CALLABLES:
        if known is callee:
            return name
    owner = type(callee)
    module = getattr(callee, "__module__", "")
    name = getattr(callee, "__qualname__", None) or getattr(callee, "__name__", "")
    if owner.__module__ == "numpy" and owner.__name__ == "ufunc":
        module = "ufunc"
    return CALLABLE_IDENTITIES.get((owner.__module__, module, name))


def op_name(callee: object) -> str:
    found = known_op(callee)
    if found is None:
        if callee is open or (
            getattr(callee, "__module__", "") in {"posix", "nt"}
            and getattr(callee, "__name__", "")
            in {
                "stat",
                "lstat",
                "open",
                "read",
                "write",
                "listdir",
                "scandir",
                "readlink",
                "access",
                "remove",
                "unlink",
                "mkdir",
                "rmdir",
                "rename",
                "replace",
                "getcwd",
                "chdir",
            }
        ):
            raise Unsupported(f"file system access: {named(callee)}")
        raise Unsupported(f"unknown callable {named(callee)}")
    return found
