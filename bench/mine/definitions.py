from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from types import FunctionType


def strip_definition(node: ast.FunctionDef | ast.Lambda) -> ast.FunctionDef | ast.Lambda:
    stripped = deepcopy(node)
    for child in ast.walk(stripped):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            child.decorator_list = []
            child.returns = None
            child.type_comment = None
        elif isinstance(child, ast.arguments):
            child.defaults = []
            child.kw_defaults = [None] * len(child.kwonlyargs)
        elif isinstance(child, ast.arg):
            child.annotation = None
            child.type_comment = None
    return stripped


def safe_definition(node: ast.FunctionDef | ast.Lambda) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and (
            child.decorator_list or child.returns is not None or child.type_params
        ):
            return False
        if isinstance(child, ast.arguments) and (
            child.defaults or any(value is not None for value in child.kw_defaults)
        ):
            return False
        if isinstance(child, ast.arg) and child.annotation is not None:
            return False
    return True


def bind_function(node: ast.FunctionDef | ast.Lambda) -> FunctionType:
    stripped = strip_definition(node)
    if not safe_definition(stripped):
        raise ValueError("definition-time expressions remain")
    name = stripped.name if isinstance(stripped, ast.FunctionDef) else "udfx_candidate"
    statement = (
        stripped
        if isinstance(stripped, ast.FunctionDef)
        else ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=stripped)
    )
    module = ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[]))
    namespace: dict[str, object] = {"__builtins__": builtins}
    exec(builtins.compile(module, "<udfx-candidate>", "exec", dont_inherit=True), namespace)
    function = namespace[name]
    assert isinstance(function, FunctionType)
    return function
