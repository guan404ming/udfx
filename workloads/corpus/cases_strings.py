# pyright: basic
from __future__ import annotations

from udfx.types import INT, STR

from . import Case
from .cases_string_splitting import CASES as SPLITTING

CODES = {"A1", "B2", "C3"}


def truthy_guard(value):
    if value:
        return value.upper()
    return None


def bare_except(s):
    try:
        return s.split("::")[1]
    except:  # noqa: E722
        return None


def split_part(project):
    if project:
        try:
            return project.split("::")[2]
        except Exception:
            return None
    return None


def has_separator(project):
    if "::" in project:
        return project.split("::")[0]
    return None


def year(date):
    if date:
        try:
            return int(date[:4])
        except Exception:
            return -1
    return None


def month(date):
    if date:
        try:
            return int(date[5:7])
        except Exception:
            return -1
    return None


def day(date):
    if date:
        try:
            return int(date[date.rfind("-") + 1 :])
        except Exception:
            return -1
    return None


def first_dash(text):
    return text.find("-")


def last_dash(text):
    return text.rfind("-")


def head_slice(text):
    return text[1:3]


def known_code(code):
    return code in CODES


def lower_strip(s):
    return s.lower().strip()


def upper(s):
    return s.upper()


def replace_dashes(s):
    return s.replace("-", "/")


def starts_with(s):
    return s.startswith("ab")


def concat(a, b):
    return a + "-" + b


def length(s):
    return len(s)


def optional_code(code):
    return code in ("A1", "B2")


def split_by(text, separator):
    return text.split(separator)


def split_on_none(text):
    return text.split(None)


def constant_receiver(s):
    return "a,b ".strip().replace(",", s)


def constant_part(i):
    try:
        return "a,b".upper().split(",")[i]
    except IndexError:
        return None


def reverse_text(text):
    return text[::-1]


def stride_text(text, start, stop):
    return text[start:stop:2]


def reverse_stride_text(text, start, stop):
    return text[start:stop:-2]


def slice_parts(text):
    return text.split("_")[:3]


def join_parts(text):
    return "_".join(text.split("_")[:3])


def stride_parts(text):
    return text.split("_")[::2]


def reverse_stride_parts(text, start, stop):
    return text.split("_")[start:stop:-2]


def index_sliced_parts(text, index):
    try:
        return text.split("_")[::-2][index]
    except IndexError:
        return None


def join_characters(text, separator):
    return separator.join(text)


def zero_slice_step(text):
    try:
        return text[::0]
    except ValueError:
        return "zero"


CASES = [
    Case("truthy_guard", truthy_guard, {"value": STR.or_none()}, "supported"),
    Case("bare_except", bare_except, {"s": STR.or_none()}, "supported"),
    Case("split_part", split_part, {"project": STR.or_none()}, "supported"),
    Case("has_separator", has_separator, {"project": STR}, "supported"),
    Case("year", year, {"date": STR.or_none()}, "supported"),
    Case("month", month, {"date": STR.or_none()}, "supported"),
    Case("day", day, {"date": STR.or_none()}, "supported"),
    Case("first_dash", first_dash, {"text": STR}, "supported"),
    Case("last_dash", last_dash, {"text": STR}, "supported"),
    Case("head_slice", head_slice, {"text": STR}, "supported"),
    Case("known_code", known_code, {"code": STR}, "supported"),
    Case("lower_strip", lower_strip, {"s": STR}, "supported"),
    Case("upper", upper, {"s": STR.or_none()}, "supported"),
    Case("replace_dashes", replace_dashes, {"s": STR}, "supported"),
    Case("starts_with", starts_with, {"s": STR}, "supported"),
    Case("concat", concat, {"a": STR, "b": STR}, "supported"),
    Case("length", length, {"s": STR}, "supported"),
    Case("optional_code", optional_code, {"code": STR.or_none()}, "supported"),
    Case("split_by", split_by, {"text": STR, "separator": STR.or_none()}, "supported"),
    Case("split_on_none", split_on_none, {"text": STR}, "supported"),
    Case("constant_receiver", constant_receiver, {"s": STR}, "supported"),
    Case("constant_part", constant_part, {"i": INT}, "supported"),
    Case("reverse_text", reverse_text, {"text": STR}, "supported"),
    Case(
        "stride_text",
        stride_text,
        {"text": STR, "start": INT.or_none(), "stop": INT.or_none()},
        "supported",
    ),
    Case(
        "reverse_stride_text",
        reverse_stride_text,
        {"text": STR, "start": INT.or_none(), "stop": INT.or_none()},
        "supported",
    ),
    Case("slice_parts", slice_parts, {"text": STR}, "supported"),
    Case("join_parts", join_parts, {"text": STR}, "supported"),
    Case("stride_parts", stride_parts, {"text": STR}, "supported"),
    Case(
        "reverse_stride_parts",
        reverse_stride_parts,
        {"text": STR, "start": INT.or_none(), "stop": INT.or_none()},
        "supported",
    ),
    Case("index_sliced_parts", index_sliced_parts, {"text": STR, "index": INT}, "supported"),
    Case("join_characters", join_characters, {"text": STR, "separator": STR}, "supported"),
    Case("zero_slice_step", zero_slice_step, {"text": STR}, "supported"),
    *SPLITTING,
]
