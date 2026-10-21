from bench.mine.models import Counts, Result
from bench.mine.report import summary


def test_summary_reports_unique_and_weighted_results() -> None:
    results = [
        Result("first", 3, "lambda value: value", ok=True, schema={"value": "str?"}),
        Result(
            "second",
            2,
            "lambda value: sum(value)",
            reason="unknown callable sum",
            category="unknown callable",
        ),
        Result(
            "third",
            1,
            "lambda value: value.casefold()",
            reason="no op .casefold(str)",
            category="no op .casefold",
        ),
    ]
    report = summary(results, Counts(notebooks=3, cells=4))
    assert "Compiled unique: 1/3 (33.33%)" in report
    assert "Compiled weighted by occurrences: 3/6 (50.00%)" in report
    assert "| unknown callable | 1 | 2 |" in report
    assert "| unknown callable sum | 1 | 2 |" in report
    assert "| no op .casefold | 1 | 1 |" in report
    assert "0/0 (0.00%)" in summary([], Counts())


def test_summary_separates_kinds_and_exclusions() -> None:
    results = [
        Result("first", 3, "lambda value: value", ok=True),
        Result(
            "second",
            2,
            "lambda row: row['value']",
            kind="row",
            reason="row schema unsupported",
            category="row schema unsupported",
        ),
        Result("third", 7, "lambda values: values", kind="group", ok=True),
    ]
    counts = Counts(data_file="input.csv", notebooks=3, cells=4)
    counts.unresolved["missing"] = 4
    counts.skipped_definitions["unsafe"] = 2
    counts.skipped_cells["parse error"] = 3
    counts.magic_cells = 1
    report = summary(results, counts)
    assert "| element + row | 1/2 (50.00%) | 3/5 (60.00%) |" in report
    assert "| element | 1/1 (100.00%) | 3/3 (100.00%) |" in report
    assert "| row | 0/1 (0.00%) | 0/2 (0.00%) |" in report
    assert "| group | 1/1 (100.00%) | 7/7 (100.00%) |" in report
    assert "out of scope for scalar UDFs" in report
    assert "Data file: input.csv" in report
    assert "Unresolved names at call sites: 4 occurrences" in report
    assert "Unsafe or unbindable definitions: 2 occurrences" in report
    assert "Parse errors: 3 cells" in report
    assert "Cells with magic or shell lines removed: 1" in report
    for kind in ("element", "row", "group"):
        assert f"Top 25 failure categories: {kind}" in report


def test_summary_excludes_third_party_names_from_both_headline_denominators() -> None:
    results = [
        Result("first", 3, "lambda value: value", ok=True),
        Result("second", 2, "lambda row: row.value", kind="row", ok=True),
        Result("third", 4, "lambda value: missing(value)", third_party_names=["missing"]),
        Result(
            "fourth", 5, "lambda row: model(row.value)", kind="row", third_party_names=["model"]
        ),
        Result("fifth", 6, "lambda value: value.expandtabs()", reason="no op .expandtabs(str)"),
        Result("sixth", 7, "lambda value: model(value)", kind="group", third_party_names=["model"]),
    ]
    report = summary(results, Counts())
    assert "### All candidates\n\n- Compiled unique: 2/5 (40.00%)" in report
    assert "Compiled weighted by occurrences: 5/20 (25.00%)" in report
    assert "### In-scope candidates\n\n- Compiled unique: 2/3 (66.67%)" in report
    assert "Compiled weighted by occurrences: 5/11 (45.45%)" in report
    assert "## Third-party names" in report
    assert "Excluded element + row candidates: 2 unique; 9 occurrences" in report
    assert "| model | 2 | 12 |" in report
    assert "| missing | 1 | 4 |" in report
    assert "X has no resolvable import and no definition in the notebook" in report
