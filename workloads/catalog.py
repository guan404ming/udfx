from collections.abc import Callable
from pathlib import Path

from workloads import flights, nyc311, udfbench
from workloads.models import Workload

SUITES: dict[str, Callable[[Path], list[Workload]]] = {
    "udfbench": udfbench.load,
    "flights": flights.load,
    "nyc311": nyc311.load,
}


def load(suites: list[str], root: Path) -> list[Workload]:
    workloads = [workload for suite in suites for workload in SUITES[suite](root)]
    identifiers = [workload.identifier for workload in workloads]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Workload identifiers must be unique")
    return workloads
