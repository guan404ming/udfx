import ast
import csv
from pathlib import Path

import pytest

from bench import mine


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
    function = mine.bind_function(node)
    assert function.__defaults__ is None
    assert function.__kwdefaults__ is None
    assert function.__annotations__ == {}
    assert set(function.__globals__) == {
        "__builtins__",
        function.__name__ if not lambda_node else "udfx_candidate",
    }
    assert mine.safe_definition(mine.strip_definition(node))
    assert ast.dump(node) == original
    mine.measure(mine.Candidate("notebook", 1, mine.strip_definition(node)))
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
        mine.bind_function(node)


@pytest.mark.parametrize(
    ("source", "sites"),
    [
        (f"frame.{name}(lambda value: value, lambda ignored: ignored)", [f".{name}"])
        for name in ("apply", "map", "applymap", "transform", "agg", "map_elements", "map_rows")
    ]
    + [
        ("udf(lambda value: value)", ["udf"]),
        ("functions.udf(function)", ["udf"]),
        ("pandas_udf(f=lambda value: value)", ["pandas_udf"]),
        ("np.vectorize(function)", ["np.vectorize"]),
        ("@udf\ndef function(value): return value", ["@udf"]),
        ("@pandas_udf(schema)\ndef function(value): return value", ["@pandas_udf"]),
        ("frame.agg(['sum'])\nframe.apply('sum')\nother.vectorize(function)", []),
    ],
)
def test_candidate_sites(source: str, sites: list[str]) -> None:
    assert [site for site, _ in mine.find_candidates(ast.parse(source))] == sites


def test_collection_groups_cells_resolves_names_and_stops_at_truncated_csv(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    rows = [
        ("first", f"open({str(marker)!r}, 'w')"),
        ("second", "frame.apply(lambda value: value.lower())"),
        ("first", "frame.map(lambda value='default': value.lower())"),
        (
            "first",
            "frame.apply(named)\nframe.apply(missing)\nframe.apply(hidden)\nframe.apply(abs)",
        ),
        ("first", "def named(value=unsafe()): return value.lower()"),
        ("other", "def missing(value): return value"),
        ("first", "def outer():\n    def hidden(value): return value"),
        ("first", "@udf(returnType=StringType())\ndef decorated(value): return value"),
        ("first", "@udf\ndef unsafe[Item](value): return value"),
        ("first", "%time frame.apply(lambda value: value)"),
        ("first", "!echo unsafe\nframe.apply(lambda value: value)"),
        ("first", "not valid python"),
        ("first", f"value = {'a' * 150_000!r}"),
    ]
    path = tmp_path / "input.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kernel_id", "cell_index", "source", "output_type", "execution_count"])
        writer.writerows(
            (kernel, index, source, "", "") for index, (kernel, source) in enumerate(rows)
        )
        stream.write('truncated,0,"unfinished')
    candidates, counts = mine.collect_candidates(path)
    assert counts.cells == len(rows)
    assert counts.notebooks == 3
    assert counts.csv_error is not None
    assert counts.unresolved == {"missing": 1, "hidden": 1, "abs": 1}
    assert counts.skipped_cells == {"IPython magic or shell line": 2, "parse error": 1}
    assert counts.skipped_definitions.total() == 1
    assert counts.call_sites == {".apply": 5, ".map": 1, "@udf": 2}
    assert len(candidates) == 3
    assert sum(candidate.occurrences for candidate in candidates) == 4
    assert candidates[0].kernel_id == "second"
    assert candidates[0].occurrences == 2
    assert not marker.exists()


@pytest.mark.parametrize(
    ("source", "schema", "category"),
    [
        ("lambda value: value.lower()", {"value": "str?"}, None),
        ("lambda value: value / 2", {"value": "float?"}, None),
        ("lambda value: 'word'[value]", {"value": "int?"}, None),
        ("lambda value: value.casefold()", None, "no op .casefold"),
        ("lambda value: sum(value)", None, "unknown callable"),
        ("lambda value: missing(value)", None, "unknown global"),
    ],
)
def test_measure_tries_schemas_and_retains_first_failure(
    source: str, schema: dict[str, str] | None, category: str | None
) -> None:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Lambda)
    result = mine.measure(mine.Candidate("notebook", 1, node))
    assert result.schema == schema
    assert result.ok == (schema is not None)
    assert result.category == category
    if category == "no op .casefold":
        assert result.reason == "no op .casefold(str)"


@pytest.mark.parametrize(
    ("reason", "category"),
    [
        ("unknown callable sum", "unknown callable"),
        ("unknown global numpy", "unknown global"),
        ("no op .casefold(str) at helper.<locals>.nested+32", "no op .casefold"),
        ("loops and comprehensions at helper+16", "loops and comprehensions"),
        ("object at 0x123ABC at helper+4", "object at <address>"),
    ],
)
def test_failure_categories(reason: str, category: str) -> None:
    assert mine.failure_category(reason) == category


def test_summary_reports_unique_and_weighted_results() -> None:
    results = [
        mine.Result("first", 3, "lambda value: value", ok=True, schema={"value": "str?"}),
        mine.Result(
            "second",
            2,
            "lambda value: sum(value)",
            reason="unknown callable sum",
            category="unknown callable",
        ),
        mine.Result(
            "third",
            1,
            "lambda value: value.casefold()",
            reason="no op .casefold(str)",
            category="no op .casefold",
        ),
    ]
    report = mine.summary(results, mine.Counts(notebooks=3, cells=4))
    assert "Compiled unique: 1/3 (33.33%)" in report
    assert "Compiled weighted by occurrences: 3/6 (50.00%)" in report
    assert "| unknown callable | 1 | 2 |" in report
    assert "| unknown callable sum | 1 | 2 |" in report
    assert "| no op .casefold | 1 | 1 |" in report
    assert "0/0 (0.00%)" in mine.summary([], mine.Counts())
