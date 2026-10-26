import pytest

from tests.frontend.test_golden import test_corpus_case_matches_the_golden_snapshot as check_case
from workloads.corpus import Case
from workloads.corpus.cases_callables import CASES


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_callable_case_matches_the_golden_snapshot(case: Case) -> None:
    check_case(case)
