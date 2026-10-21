# pyright: basic
from __future__ import annotations

import json

from udfx.types import INT, STR, list_of

from . import Case


def is_none(x):
    return x is None


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


def parse_json(s):
    return json.loads(s)


def first_or_default(items):
    if items:
        first = items[0]
        if first is not None:
            return first
    return "default"


def join_or_error(items):
    return ",".join(items)


def keyword_split(s):
    return s.split(sep="::")


def keyword_int(s):
    return int(s, base=16)


CASES = [
    Case("is_none", is_none, {"x": INT.or_none()}, "native"),
    Case("nested_helper", nested_helper, {"s": STR.or_none()}, "native"),
    Case("conditional_expression", conditional_expression, {"n": INT}, "native"),
    Case(
        "first_or_default", first_or_default, {"items": list_of(STR.or_none()).or_none()}, "native"
    ),
    Case("join_or_error", join_or_error, {"items": list_of(STR.or_none())}, "native"),
    Case("loop_total", loop_total, {"s": STR}, "fallback"),
    Case("parse_json", parse_json, {"s": STR}, "fallback"),
    Case("keyword_split", keyword_split, {"s": STR}, "fallback"),
    Case("keyword_int", keyword_int, {"s": STR}, "fallback"),
]
