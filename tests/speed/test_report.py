from bench.speed.report import report
from bench.speed.timing import Measurement


def test_report_thresholds_and_skips() -> None:
    output = report(
        [
            Measurement("slow", "polars eager", 0.5, 1.0, "map_elements", True, "first", 0.0),
            Measurement("fast", "polars eager", 32.0, 1.0, "map_elements", False, "second", 0.0),
            Measurement("equal", "duckdb", 1.0, 1.0, "native", False, "first", 0.0),
        ],
        {"example / spark": "engine unavailable"},
        200,
        3,
    )
    assert (
        "| slow | polars eager | map_elements | 0.500000 | 1.000000 | 0.50x | true | 0.00% | FAIL |"
        in output
    )
    assert (
        "| equal | duckdb | native | 1.000000 | 1.000000 | 1.00x | false | 0.00% | PASS |" in output
    )
    assert "| polars eager | 2 | 2 | 1 | 4.00x | 4.00x | PASS | 1/2 |" in output
    assert "| duckdb | 1 | 1 | 0 | 1.00x | 1.00x | FAIL | 1/1 |" in output
    assert "skipped example / spark: engine unavailable" in output


def test_report_deduplicates_per_engine() -> None:
    output = report(
        [
            Measurement("slow", "polars eager", 0.125, 1.0, "map_elements", True, "unique", 0.0),
            Measurement("fast", "polars eager", 8.0, 1.0, "map_elements", False, "shared", 0.0),
            Measurement("faster", "polars eager", 128.0, 1.0, "map_elements", False, "shared", 0.0),
            Measurement("other", "duckdb", 3.0, 1.0, "native", False, "shared", 0.0),
        ],
        {},
        200,
        3,
    )
    assert "| polars eager | 3 | 2 | 1 | 5.04x | 2.00x | FAIL | 2/3 |" in output
    assert "| duckdb | 1 | 1 | 0 | 3.00x | 3.00x | PASS | 1/1 |" in output
    assert (
        "| slow | polars eager | map_elements | 0.125000 | 1.000000 | 0.12x | true | 0.00% | FAIL |"
        in output
    )
    duplicates = output.split("\n\n")[3].splitlines()
    assert duplicates == [
        "| engine | fingerprint | cases | speedups |",
        "|---|---|---|---|",
        "| polars eager | a4d26868017c | fast, faster | 8.00x, 128.00x |",
    ]
