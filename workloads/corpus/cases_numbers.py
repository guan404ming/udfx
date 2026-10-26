# pyright: basic
from __future__ import annotations

import math
import random

from udfx.types import FLOAT, INT, STR

from . import Case

EURO_RATES = {"EUR": 1.0, "USD": 1.09, "GBP": 0.85, "CHF": 0.98}


def int_above_one(x):
    return int(x) > 1


def inverse(x):
    return 1 / x


def truthy_int_guard(value):
    if value:
        return 100 // value
    return None


def try_default(s):
    try:
        return int(s)
    except Exception:
        return -1


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


def compound(principal, rate):
    return principal * (1.0 + rate) ** 12.0


def nearest_whole(x):
    if math.isnan(x) or math.isinf(x):
        return 0
    return round(x)


def clamp(x):
    return max(0.0, min(x, 1.0))


def gauss_noise(x):
    return x + random.gauss(0, 1)


def text_or_empty(x):
    try:
        return str(x)
    except TypeError:
        return ""


def float_text(x):
    return str(x)


def is_zero(x):
    return x == 0


def not_half(a):
    return a != 1.5


CASES = [
    Case("int_above_one", int_above_one, {"x": STR}, "supported"),
    Case("inverse", inverse, {"x": INT}, "supported"),
    Case("truthy_int_guard", truthy_int_guard, {"value": INT.or_none()}, "supported"),
    Case("try_default", try_default, {"s": STR}, "supported"),
    Case("euro", euro, {"amount": FLOAT.or_none(), "currency": STR.or_none()}, "supported"),
    Case("log_ten", log_ten, {"x": FLOAT.or_none()}, "supported"),
    Case("int_arithmetic", int_arithmetic, {"a": INT, "b": INT}, "supported"),
    Case("int_floor_division", int_floor_division, {"a": INT, "b": INT}, "supported"),
    Case("int_modulo", int_modulo, {"a": INT, "b": INT}, "supported"),
    Case("float_division", float_division, {"a": FLOAT, "b": FLOAT}, "supported"),
    Case("float_arithmetic", float_arithmetic, {"a": FLOAT, "b": FLOAT}, "supported"),
    Case("safe_division", safe_division, {"a": INT, "b": INT}, "supported"),
    Case("compound", compound, {"principal": FLOAT, "rate": FLOAT}, "supported"),
    Case("nearest_whole", nearest_whole, {"x": FLOAT}, "supported"),
    Case("clamp", clamp, {"x": FLOAT}, "supported"),
    Case("text_or_empty", text_or_empty, {"x": INT.or_none()}, "supported"),
    Case("float_text", float_text, {"x": FLOAT.or_none()}, "supported"),
    Case("is_zero", is_zero, {"x": FLOAT}, "supported"),
    Case("not_half", not_half, {"a": INT.or_none()}, "supported"),
    Case("gauss_noise", gauss_noise, {"x": FLOAT}, "optional"),
]
