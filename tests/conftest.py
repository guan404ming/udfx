import os
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from hypothesis import HealthCheck, settings

from udfx.backends import installed

TESTS = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [TESTS, os.environ.get("PYTHONPATH")]))

settings.register_profile(
    "default", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def backend_params() -> list[ParameterSet]:
    return [
        pytest.param(backend, id=backend.name, marks=pytest.mark.xdist_group(backend.name))
        for backend in installed()
    ]
