# pyright: basic
# ruff: noqa: PLW0108
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


def row_positions(row):
    return row[0] + ":" + str(row[-1])


def row_position_fallback(row):
    return str(int(row[0])) + row[-1]


def get_day_str(row):
    return row["day"].strip().lower()


def make_path(row):
    def filename(record):
        return record.image + ".jpg"

    return row["directory"] + "/" + filename(row)


def week_function(row):
    feature = "stays_in_week_nights"
    return row[feature] > 0


CASES = [
    Case("row_total", row_total, {"quantity": INT, "price": FLOAT}, "native"),
    Case("row_image", row_image, {"image": STR.or_none()}, "native"),
    Case("row_year", row_year, {"offset": INT, "date": STR.or_none()}, "native"),
    Case("row_helper", row_helper, {"image": STR}, "native"),
    Case("row_positions", row_positions, {"text": STR, "number": INT}, "native"),
    Case("row_position_fallback", row_position_fallback, {"number": FLOAT, "text": STR}, "native"),
    Case("row_day_helper", lambda row: get_day_str(row), {"day": STR}, "native"),
    Case(
        "row_path_helper",
        lambda row: make_path(row),
        {"directory": STR, "image": STR},
        "native",
    ),
    Case("row_week_local_key", week_function, {"stays_in_week_nights": INT}, "native"),
    Case(
        "row_position_helper_fallback",
        lambda row: row_position_fallback(row),
        {"number": FLOAT, "text": STR},
        "native",
    ),
]
