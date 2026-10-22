from bench.speed.report import report
from bench.speed.timing import Measurement


def test_report_thresholds_and_skips() -> None:
    output = report(
        [
            Measurement("slow", "polars eager", 0.5, 1.0, "map_elements", True),
            Measurement("fast", "polars eager", 32.0, 1.0, "map_elements", False),
            Measurement("equal", "duckdb", 1.0, 1.0, "native", False),
        ],
        {"example / spark": "engine unavailable"},
        200,
        3,
    )
    assert (
        "| slow | polars eager | map_elements | 0.500000 | 1.000000 | 0.50x | true | FAIL |"
        in output
    )
    assert "| equal | duckdb | native | 1.000000 | 1.000000 | 1.00x | false | PASS |" in output
    assert "| polars eager | 2 | 1 | 4.00x | PASS | 1/2 |" in output
    assert "| duckdb | 1 | 0 | 1.00x | FAIL | 1/1 |" in output
    assert "skipped example / spark: engine unavailable" in output
