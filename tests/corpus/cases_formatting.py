# ruff: noqa: UP031, UP032
from udfx.types import BOOL, FLOAT, INT, STR

from . import Case


def f_string_record(text: str | None, number: int | None, flag: bool | None) -> str:
    return f"{{{text!s}}}:{number!r}:{flag}"


def f_string_float(value: float | None) -> str:
    return f"{value}:{value!r}"


def f_string_specifiers(text: str | None, number: int | None) -> str:
    try:
        return f"{text:s}:{number:d}"
    except TypeError:
        return "missing"


def format_automatic(text: str | None, number: int | None) -> str:
    return "{} {}".format(text, number)


def format_indexed(text: str | None, number: int | None) -> str:
    return "{{{1!r:s}}}-{0!s}-{1}".format(text, number)


def format_specifiers(text: str | None, number: int | None) -> str:
    try:
        return "{:s}:{:d}".format(text, number)
    except TypeError:
        return "missing"


def percent_string(value: float | None) -> str:
    return "%s" % value


def percent_pair(text: str | None, number: int | None) -> str:
    return "%s-%s" % (text, number)


def percent_integer(text: str | None, number: int | None) -> str:
    try:
        return "%s-%d%%" % (text, number)
    except TypeError:
        return "missing"


CASES = [
    Case(
        "f_string_record",
        f_string_record,
        {"text": STR.or_none(), "number": INT.or_none(), "flag": BOOL.or_none()},
        "native",
    ),
    Case("f_string_float", f_string_float, {"value": FLOAT.or_none()}, "native"),
    Case(
        "f_string_specifiers",
        f_string_specifiers,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
    Case(
        "format_automatic",
        format_automatic,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
    Case(
        "format_indexed",
        format_indexed,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
    Case(
        "format_specifiers",
        format_specifiers,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
    Case("percent_string", percent_string, {"value": FLOAT.or_none()}, "native"),
    Case(
        "percent_pair",
        percent_pair,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
    Case(
        "percent_integer",
        percent_integer,
        {"text": STR.or_none(), "number": INT.or_none()},
        "native",
    ),
]
