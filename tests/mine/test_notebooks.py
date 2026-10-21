import csv
from pathlib import Path

from bench.mine.notebooks import collect_candidates

from .helpers import collect_sources


def test_collection_counts_each_kind_once(tmp_path: Path) -> None:
    candidates, _ = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "def function(value): return value\n"
                "frame.map(function)\nframe.apply(function)\n"
                "frame.apply(function, axis=1)\nframe.agg(function)",
            )
        ],
    )
    assert {candidate.kind: candidate.occurrences for candidate in candidates} == {
        "element": 2,
        "row": 1,
        "group": 1,
    }


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
    candidates, counts = collect_candidates(path)
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


def test_collection_distinguishes_referenced_globals(tmp_path: Path) -> None:
    candidates, _ = collect_sources(
        tmp_path,
        [
            ("first", "offset = 1\nframe.map(lambda value: value + offset)"),
            ("second", "offset = 2\nframe.map(lambda value: value + offset)"),
            ("third", "offset = 1\nunused = 5\nframe.map(lambda value: value + offset)"),
        ],
    )
    assert len(candidates) == 2
    assert [candidate.occurrences for candidate in candidates] == [2, 1]
