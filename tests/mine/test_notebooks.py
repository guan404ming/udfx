import csv
from pathlib import Path

import pytest

from bench.mine.bindings import bind_candidate
from bench.mine.measurement import measure
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
    assert counts.skipped_cells == {"parse error": 1}
    assert counts.magic_cells == 2
    assert counts.skipped_definitions.total() == 1
    assert counts.call_sites == {".apply": 6, ".map": 1, "@udf": 2}
    assert len(candidates) == 4
    assert sum(candidate.occurrences for candidate in candidates) == 5
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


@pytest.mark.parametrize("directive", ["%matplotlib inline", "  !echo ignored", "%%time"])
def test_magic_cells_preserve_imports_helpers_and_candidates(
    tmp_path: Path, directive: str
) -> None:
    marker = tmp_path / "executed"
    candidates, counts = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                f"{directive}\nimport re\n"
                "def helper(value): return re.sub('a', 'b', value)\n"
                f"open({str(marker)!r}, 'w')\n"
                "frame.map(lambda value: helper(value))",
            )
        ],
    )
    assert counts.magic_cells == 1
    assert not counts.skipped_cells
    assert len(candidates) == 1
    assert set(candidates[0].bindings) == {"helper", "re"}
    assert measure(candidates[0]).ok
    assert not marker.exists()


def test_shell_continuations_do_not_become_python_bindings(tmp_path: Path) -> None:
    candidates, counts = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "!echo first \\\n"
                "    --second \\\n"
                "offset = 99\n"
                "import math\n"
                "frame.map(lambda value: math.sqrt(value) + offset)",
            )
        ],
    )
    assert not counts.skipped_cells
    assert len(candidates) == 1
    namespace = bind_candidate(candidates[0]).__globals__
    assert "math" in namespace
    assert "offset" not in namespace


def test_magic_removal_preserves_multiline_string_contents(tmp_path: Path) -> None:
    candidates, counts = collect_sources(
        tmp_path,
        [
            (
                "notebook",
                "%matplotlib inline\n"
                "suffix = '''\n%literal\n!literal\n'''\n"
                "frame.map(lambda value: value + suffix)",
            )
        ],
    )
    assert counts.magic_cells == 1
    assert bind_candidate(candidates[0]).__globals__["suffix"] == "\n%literal\n!literal\n"
