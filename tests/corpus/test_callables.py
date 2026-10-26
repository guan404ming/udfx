from typing import Any

import pytest
from conftest import backend_params

from udfx.backends.base import Backend
from workloads.corpus.cases_callables import CASES, Case

from .test_corpus import test_case as check_case

pytestmark = pytest.mark.corpus


@pytest.mark.parametrize("backend", backend_params())
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_callable_case(case: Case, backend: Backend[Any]) -> None:
    check_case(case, backend)
