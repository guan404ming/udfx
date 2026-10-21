# pyright: basic
from udfx.types import FLOAT, INT, STR

from . import Case


def row_total(row):
    return row["price"] * row["quantity"]


def row_image(row):
    return row["image"].strip().lower() if row["image"] else None


def row_year(row):
    try:
        return int(row.date[:4]) + row.offset
    except (ValueError, TypeError):
        return -1


def row_helper(row):
    def clean(value):
        return value.strip().lower()

    return clean(row.image)


CASES = [
    Case("row_total", row_total, {"quantity": INT, "price": FLOAT}, "native"),
    Case("row_image", row_image, {"image": STR.or_none()}, "native"),
    Case("row_year", row_year, {"offset": INT, "date": STR.or_none()}, "native"),
    Case("row_helper", row_helper, {"image": STR}, "native"),
]
