# pyright: basic
# ruff: noqa: E721, SIM401
from __future__ import annotations

import calendar
import json

from udfx.types import BOOL, FLOAT, INT, STR, list_of

from . import Case


def is_none(x):
    return x is None


def nested_helper(s):
    def clean(part):
        return part.strip().lower()

    if s:
        return clean(s)
    return None


def clean_text(text, suffix="!"):
    return text.strip().lower() + suffix


def global_helper(text):
    return clean_text(text)


def helper_chain(text):
    return global_helper(text)


def keyword_helper(text):
    return clean_text(suffix="?", text=text)


def library_helper(year):
    return calendar.isleap(year)


def keyword_round(value):
    return round(value, ndigits=2)


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


def type_identity(value):
    return type(value) is str


def type_equality(value):
    return type(value) == str


def instance_text(value):
    return isinstance(value, str)


def instance_number(value):
    return isinstance(value, (int, float))


def dictionary_default(value):
    return {"a": "b", "c": "d"}.get(value, value)


def dictionary_none(value):
    return {"a": "b", "c": "d"}.get(value)


def dictionary_guard(value):
    mapping = {"a": "b", "c": "d"}
    return mapping[value] if value in mapping else value


STOPWORDS = {"a", "the"}


def remove_stopwords(text):
    def clean(value, *, stop=STOPWORDS):
        return " ".join(word for word in value.split() if word not in stop)

    return clean(text)


def default_mapping(text):
    def clean(value, mapping={"a": "alpha", "b": "beta"}):  # noqa: B006
        return mapping.get(value, value)

    return clean(text)


CASES = [
    Case("container_default_stopwords", remove_stopwords, {"text": STR}, "native"),
    Case("container_default_mapping", default_mapping, {"text": STR}, "native"),
    Case("type_identity", type_identity, {"value": STR.or_none()}, "native"),
    Case("type_equality", type_equality, {"value": STR.or_none()}, "native"),
    Case("instance_text", instance_text, {"value": STR.or_none()}, "native"),
    Case("instance_number", instance_number, {"value": FLOAT.or_none()}, "native"),
    Case("instance_boolean", instance_number, {"value": BOOL.or_none()}, "native"),
    Case("dictionary_default", dictionary_default, {"value": STR.or_none()}, "native"),
    Case("dictionary_none", dictionary_none, {"value": STR.or_none()}, "native"),
    Case("dictionary_guard", dictionary_guard, {"value": STR.or_none()}, "native"),
    Case("is_none", is_none, {"x": INT.or_none()}, "native"),
    Case("nested_helper", nested_helper, {"s": STR.or_none()}, "native"),
    Case("global_helper", global_helper, {"text": STR.or_none()}, "native"),
    Case("helper_chain", helper_chain, {"text": STR.or_none()}, "native"),
    Case("keyword_helper", keyword_helper, {"text": STR.or_none()}, "native"),
    Case("library_helper", library_helper, {"year": INT}, "native"),
    Case("keyword_round", keyword_round, {"value": FLOAT}, "native"),
    Case("conditional_expression", conditional_expression, {"n": INT}, "native"),
    Case(
        "first_or_default", first_or_default, {"items": list_of(STR.or_none()).or_none()}, "native"
    ),
    Case("join_or_error", join_or_error, {"items": list_of(STR.or_none())}, "native"),
    Case("loop_total", loop_total, {"s": STR}, "fallback"),
    Case("parse_json", parse_json, {"s": STR}, "fallback"),
    Case("keyword_split", keyword_split, {"s": STR}, "native"),
    Case("keyword_int", keyword_int, {"s": STR}, "fallback"),
]
