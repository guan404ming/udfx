from typing import Any


def inverse(x: int) -> float:
    return 1 / x


def year(text: str) -> int:
    return int(text[:4]) + 1


def row_year(row: Any) -> int:
    return int(row["text"]) + row.offset


def row_inverse(row: Any) -> float:
    return 1 / int(row.value)
