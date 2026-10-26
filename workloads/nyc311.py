from datetime import datetime
from pathlib import Path

from udfx.types import DATETIME, FLOAT, STR
from workloads.models import Workload


def postal_code(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) != 5 or not value.isascii() or not value.isdigit() or value == "00000":
        return None
    return value


def complaint_family(complaint: str | None) -> str:
    if complaint is None:
        return "unknown"
    complaint = complaint.lower()
    if "noise" in complaint:
        return "noise"
    if "heat" in complaint or "water" in complaint:
        return "housing"
    return "other"


def borough_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().title()


def resolution_hours(created: datetime | None, closed: datetime | None) -> float | None:
    if created is None or closed is None:
        return None
    return (closed - created).total_seconds() / 3600.0


def load(root: Path) -> list[Workload]:
    table = root / "nyc311" / "requests.parquet"
    return [
        Workload(
            "nyc311",
            "postal_code",
            postal_code,
            {"value": STR.or_none()},
            STR.or_none(),
            table,
            {"value": "incident_zip"},
            "agency",
        ),
        Workload(
            "nyc311",
            "complaint_family",
            complaint_family,
            {"complaint": STR.or_none()},
            STR,
            table,
            {"complaint": "complaint_type"},
            "agency",
        ),
        Workload(
            "nyc311",
            "borough_name",
            borough_name,
            {"value": STR.or_none()},
            STR.or_none(),
            table,
            {"value": "borough"},
            "agency",
        ),
        Workload(
            "nyc311",
            "resolution_hours",
            resolution_hours,
            {"created": DATETIME.or_none(), "closed": DATETIME.or_none()},
            FLOAT.or_none(),
            table,
            {"created": "created_date", "closed": "closed_date"},
            "agency",
        ),
    ]
