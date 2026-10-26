from udfx.backends.datafusion import DataFusionBackend
from udfx.backends.polars import PolarsBackend
from udfx.frontend import trace
from udfx.types import STR, list_of
from workloads.corpus.cases_small_gaps import CASES


def test_polars_case_chain_checks_unicode_once() -> None:
    case = next(case for case in CASES if case.name == "small_case_mapping")
    program = trace(case.function, case.schema)
    backend = PolarsBackend()
    lowered = backend.lower_program(program)
    assert lowered.exact is not None
    assert str(lowered.exact).count("str.contains") == 1
    rows = [(value,) for value in ("abc", "École", "\u0131stanbul", "中文ABC", "Σ", "ß")]
    assert backend.evaluate([lowered.exact], case.schema, rows) == [
        [True, True, True, True, False, False]
    ]


def test_datafusion_empty_union_keeps_reusable_cardinality() -> None:
    def empty(left: list[str], right: list[str]) -> bool:
        return len(set(left) | set(right)) == 0

    program = trace(empty, {"left": list_of(STR), "right": list_of(STR)})
    expression = str(DataFusionBackend().lower(program.value).expr)
    assert "cardinality(array_union(" in expression
