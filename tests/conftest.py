import os
from pathlib import Path

from hypothesis import HealthCheck, settings

TESTS = str(Path(__file__).resolve().parent)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [TESTS, os.environ.get("PYTHONPATH")]))

settings.register_profile(
    "default", max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("ci", max_examples=300, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
