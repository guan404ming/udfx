# pyright: basic
from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from udfx.types import FLOAT, INT, STR, Type


@dataclass(frozen=True)
class Case:
    name: str
    function: Callable[..., object]
    schema: Mapping[str, Type]
    expect: Literal["native", "fallback"]


EURO_RATES = {"EUR": 1.0, "USD": 1.09, "GBP": 0.85, "CHF": 0.98}
CODES = {"A1", "B2", "C3"}


def is_none(x):
    return x is None


def int_above_one(x):
    return int(x) > 1


def inverse(x):
    return 1 / x


def truthy_guard(value):
    if value:
        return value.upper()
    return None


def truthy_int_guard(value):
    if value:
        return 100 // value
    return None


def try_default(s):
    try:
        return int(s)
    except Exception:
        return -1


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


def euro(amount, currency):
    if amount is not None and currency is not None:
        try:
            return float(amount) / EURO_RATES[currency]
        except Exception:
            return 0.0
    return None


def log_ten(x):
    try:
        return math.log10(x)
    except Exception:
        return None


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


def int_arithmetic(a, b):
    return (a + b) * 2 - a


def int_floor_division(a, b):
    return a // b


def int_modulo(a, b):
    return a % b


def float_division(a, b):
    return a / b


def float_arithmetic(a, b):
    return a * b + 1.5


def safe_division(a, b):
    try:
        return a // b
    except ZeroDivisionError:
        return 0


def nested_helper(s):
    def clean(part):
        return part.strip().lower()

    if s:
        return clean(s)
    return None


def conditional_expression(n):
    return "big" if n > 100 else "small"


def loop_total(s):
    total = 0
    for char in s:
        total += ord(char)
    return total


def gauss_noise(x):
    return x + random.gauss(0, 1)


def parse_json(s):
    return json.loads(s)


def keyword_split(s):
    return s.split(sep="::")


def keyword_int(s):
    return int(s, base=16)


CASES = [
    Case("is_none", is_none, {"x": INT.or_none()}, "native"),
    Case("int_above_one", int_above_one, {"x": STR}, "native"),
    Case("inverse", inverse, {"x": INT}, "native"),
    Case("truthy_guard", truthy_guard, {"value": STR.or_none()}, "native"),
    Case("truthy_int_guard", truthy_int_guard, {"value": INT.or_none()}, "native"),
    Case("try_default", try_default, {"s": STR}, "native"),
    Case("bare_except", bare_except, {"s": STR.or_none()}, "native"),
    Case("split_part", split_part, {"project": STR.or_none()}, "native"),
    Case("has_separator", has_separator, {"project": STR}, "native"),
    Case("year", year, {"date": STR.or_none()}, "native"),
    Case("month", month, {"date": STR.or_none()}, "native"),
    Case("day", day, {"date": STR.or_none()}, "native"),
    Case("first_dash", first_dash, {"text": STR}, "native"),
    Case("last_dash", last_dash, {"text": STR}, "native"),
    Case("head_slice", head_slice, {"text": STR}, "native"),
    Case("euro", euro, {"amount": FLOAT.or_none(), "currency": STR.or_none()}, "native"),
    Case("log_ten", log_ten, {"x": FLOAT.or_none()}, "native"),
    Case("known_code", known_code, {"code": STR}, "native"),
    Case("lower_strip", lower_strip, {"s": STR}, "native"),
    Case("upper", upper, {"s": STR.or_none()}, "native"),
    Case("replace_dashes", replace_dashes, {"s": STR}, "native"),
    Case("starts_with", starts_with, {"s": STR}, "native"),
    Case("concat", concat, {"a": STR, "b": STR}, "native"),
    Case("length", length, {"s": STR}, "native"),
    Case("int_arithmetic", int_arithmetic, {"a": INT, "b": INT}, "native"),
    Case("int_floor_division", int_floor_division, {"a": INT, "b": INT}, "native"),
    Case("int_modulo", int_modulo, {"a": INT, "b": INT}, "native"),
    Case("float_division", float_division, {"a": FLOAT, "b": FLOAT}, "native"),
    Case("float_arithmetic", float_arithmetic, {"a": FLOAT, "b": FLOAT}, "native"),
    Case("safe_division", safe_division, {"a": INT, "b": INT}, "native"),
    Case("nested_helper", nested_helper, {"s": STR.or_none()}, "native"),
    Case("conditional_expression", conditional_expression, {"n": INT}, "native"),
    Case("loop_total", loop_total, {"s": STR}, "fallback"),
    Case("gauss_noise", gauss_noise, {"x": FLOAT}, "fallback"),
    Case("parse_json", parse_json, {"s": STR}, "fallback"),
    Case("keyword_split", keyword_split, {"s": STR}, "fallback"),
    Case("keyword_int", keyword_int, {"s": STR}, "fallback"),
]
