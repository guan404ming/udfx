import ast
from pathlib import Path

import pytest

from bench.mine.definitions import bind_function, safe_definition, strip_definition
from bench.mine.measurement import measure
from bench.mine.models import Candidate


@pytest.mark.parametrize("lambda_node", [False, True])
def test_binding_never_executes_definition_expressions_or_body(
    tmp_path: Path, lambda_node: bool
) -> None:
    marker = tmp_path / "executed"
    trigger = f"open({str(marker)!r}, 'w')"
    source = (
        f"lambda value={trigger}: {trigger}"
        if lambda_node
        else (
            f"@{trigger}\n"
            f"def candidate(value: {trigger} = {trigger}, /, *items: {trigger}, "
            f"option: {trigger} = {trigger}, **keywords: {trigger}) -> {trigger}:\n"
            f"    @{trigger}\n"
            f"    def nested(item: {trigger} = {trigger}) -> {trigger}:\n"
            f"        return {trigger}\n"
            f"    @{trigger}\n"
            f"    async def asynchronous(item: {trigger} = {trigger}) -> {trigger}:\n"
            f"        return {trigger}\n"
            f"    helper = lambda item={trigger}: {trigger}\n"
            f"    return {trigger}\n"
        )
    )
    statement = ast.parse(source).body[0]
    node = statement.value if isinstance(statement, ast.Expr) else statement
    assert isinstance(node, ast.FunctionDef | ast.Lambda)
    original = ast.dump(node)
    function = bind_function(node)
    assert function.__defaults__ is None
    assert function.__kwdefaults__ is None
    assert function.__annotations__ == {}
    assert set(function.__globals__) == {
        "__builtins__",
        function.__name__ if not lambda_node else "udfx_candidate",
    }
    assert safe_definition(strip_definition(node))
    assert ast.dump(node) == original
    measure(Candidate("notebook", 1, strip_definition(node)))
    assert not marker.exists()


@pytest.mark.parametrize(
    "source",
    [
        "def candidate[Item: print('unsafe')](value): return value",
        "def candidate(value):\n"
        "    def nested[Item: print('unsafe')](value): return value\n"
        "    return value",
    ],
)
def test_binding_rejects_remaining_definition_expressions(source: str) -> None:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    with pytest.raises(ValueError, match="definition-time expressions remain"):
        bind_function(node)
