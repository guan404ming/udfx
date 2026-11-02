import ast
import builtins
import symtable
from collections.abc import Callable, Mapping
from copy import deepcopy
from importlib.util import decode_source
from pathlib import Path
from typing import cast

MODULES = {"datetime", "math", "numpy", "pandas"}
SPARK_TYPES = {"BooleanType", "DoubleType", "IntegerType", "StringType"}
DYNAMIC_NAMES = {"eval", "exec", "globals"}
EXPRESSIONS = (
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.BinOp,
    ast.BitOr,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Load,
)


def source_bindings(module: ast.Module) -> dict[str, ast.stmt | None]:
    result: dict[str, ast.stmt | None] = {}
    for statement in module.body:
        entries = {}
        if isinstance(statement, ast.Import | ast.ImportFrom):
            for alias in statement.names:
                node = deepcopy(statement)
                node.names = [alias]
                name = alias.asname or alias.name.split(".")[0]
                entries[name] = node
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            entries[statement.name] = statement
        else:
            entries = {
                node.id: statement
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
        for name, node in entries.items():
            if name in result:
                previous = result[name]
                if (
                    isinstance(node, ast.Import | ast.ImportFrom)
                    and previous is not None
                    and ast.dump(node) == ast.dump(previous)
                ):
                    continue
                result[name] = None
            else:
                result[name] = node
    return result


def qualified(node: ast.expr, available: dict[str, ast.stmt | None]) -> str:
    if isinstance(node, ast.Attribute):
        return f"{qualified(node.value, available)}.{node.attr}"
    if isinstance(node, ast.Name):
        statement = available.get(node.id)
        if isinstance(statement, ast.ImportFrom) and not statement.level:
            return f"{statement.module}.{statement.names[0].name}"
        if isinstance(statement, ast.Import):
            alias = statement.names[0]
            return alias.name if alias.asname else alias.name.split(".")[0]
    return ""


def spark_wrapper(node: ast.expr, available: dict[str, ast.stmt | None]) -> bool:
    if qualified(node, available) == "pyspark.sql.functions.udf":
        return True
    if not isinstance(node, ast.Call):
        return False
    if qualified(node.func, available) != "pyspark.sql.functions.udf":
        return False
    arguments = list(node.args)
    for keyword in node.keywords:
        if keyword.arg != "returnType":
            return False
        arguments.append(keyword.value)
    if len(arguments) != 1:
        return False
    argument = arguments[0]
    return (
        isinstance(argument, ast.Call)
        and not argument.args
        and not argument.keywords
        and qualified(argument.func, available)
        in {f"pyspark.sql.types.{name}" for name in SPARK_TYPES}
    )


def expression_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if any(not isinstance(child, EXPRESSIONS) for child in ast.walk(node)):
        raise ValueError("Unsupported definition-time expression")
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def global_names(table: symtable.SymbolTable) -> set[str]:
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and symbol.is_referenced()
    }
    for child in table.get_children():
        names.update(global_names(child))
    return names


def definition_names(node: ast.FunctionDef, annotations: bool) -> set[str]:
    if node.type_params:
        raise ValueError("Generic function definitions are unsupported")
    names = set()
    for value in [*node.args.defaults, *node.args.kw_defaults]:
        names.update(expression_names(value))
    if not annotations:
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments.extend(value for value in (node.args.vararg, node.args.kwarg) if value)
        for argument in arguments:
            names.update(expression_names(argument.annotation))
        names.update(expression_names(node.returns))
    return names


def select_binding(
    name: str,
    available: dict[str, ast.stmt | None],
    symbols: symtable.SymbolTable,
    annotations: bool,
) -> tuple[ast.stmt, set[str]]:
    statement = available[name]
    dependencies = set()
    if isinstance(statement, ast.FunctionDef):
        if len(statement.decorator_list) > 1 or not all(
            spark_wrapper(value, available) for value in statement.decorator_list
        ):
            raise ValueError(f"Unsupported decorator: {name}")
        table = next(
            child
            for child in symbols.get_children()
            if child.get_name() == name and child.get_lineno() == statement.lineno
        )
        dependencies = global_names(table) | definition_names(statement, annotations)
        for dependency in dependencies:
            target = available.get(dependency)
            if isinstance(target, ast.FunctionDef) and target.decorator_list:
                raise ValueError(f"Decorated dependency is unsupported: {dependency}")
        statement = deepcopy(statement)
        statement.decorator_list = []
    elif isinstance(statement, ast.Import | ast.ImportFrom):
        imported = (
            statement.names[0].name if isinstance(statement, ast.Import) else statement.module
        )
        if imported not in MODULES or (isinstance(statement, ast.ImportFrom) and statement.level):
            raise ValueError(f"Unsupported source import: {name}")
    elif isinstance(statement, ast.Assign | ast.AnnAssign):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not all(isinstance(target, ast.Name) for target in targets):
            raise ValueError(f"Unsupported source assignment: {name}")
        if statement.value is None:
            raise ValueError(f"Missing source value: {name}")
        try:
            ast.literal_eval(statement.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Nonliteral source binding: {name}") from error
        if isinstance(statement, ast.AnnAssign) and not annotations:
            dependencies = expression_names(statement.annotation)
    else:
        raise ValueError(f"Unsupported or ambiguous source binding: {name}")
    return statement, dependencies


def check_ignored(module: ast.Module, selected: dict[str, ast.stmt]) -> None:
    positions = {(node.lineno, node.col_offset) for node in selected.values()}
    for statement in module.body:
        if isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.Import | ast.ImportFrom
        ):
            continue
        if (statement.lineno, statement.col_offset) in positions:
            continue
        references = {node.id for node in ast.walk(statement) if isinstance(node, ast.Name)}
        if references & selected.keys():
            raise ValueError("Top-level statement depends on extracted bindings")


def load_function(
    path: Path, name: str, *, bindings: Mapping[str, object] | None = None
) -> Callable[..., object]:
    source = decode_source(path.read_bytes())
    module = ast.parse(source, filename=str(path))
    symbols = symtable.symtable(source, str(path), "exec")
    if any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        for node in module.body
    ):
        raise ValueError("Wildcard source imports are unsupported")
    available = source_bindings(module)
    additions = dict(bindings or {})
    if additions.keys() & (available.keys() | vars(builtins).keys()):
        raise ValueError("Supplied bindings must not replace source or builtin names")
    if not isinstance(available.get(name), ast.FunctionDef):
        raise ValueError(f"Missing or ambiguous function: {name}")
    futures = [
        node
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    annotations = any(alias.name == "annotations" for node in futures for alias in node.names)
    selected: dict[str, ast.stmt] = {}
    imported = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in selected:
            continue
        if current not in available:
            if current in additions:
                imported.add(current)
                continue
            if current in DYNAMIC_NAMES:
                raise ValueError(f"Dynamic source namespace is unsupported: {current}")
            if current in vars(builtins):
                continue
            raise ValueError(f"Missing source binding: {current}")
        statement, dependencies = select_binding(current, available, symbols, annotations)
        selected[current] = statement
        pending.extend(sorted(dependencies - selected.keys()))
    if imported != additions.keys():
        raise ValueError("Supplied source bindings must all be used")
    check_ignored(module, selected)
    statements = sorted(
        {id(node): node for node in selected.values()}.values(), key=lambda node: node.lineno
    )
    extracted = ast.Module(body=[*futures, *statements], type_ignores=module.type_ignores)
    namespace: dict[str, object] = {"__name__": path.stem, "__file__": str(path), **additions}
    exec(builtins.compile(extracted, str(path), "exec", dont_inherit=True), namespace)
    return cast(Callable[..., object], namespace[name])
