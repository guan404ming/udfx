import re

from udfx.types import STR

from . import Case


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)


def keep_digits(text: str) -> str:
    return re.compile(r"\D+").sub("", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def first_number(text: str) -> str:
    found = re.search(r"\d+", text)
    if found is not None:
        return found.group(0)
    return ""


CASES = [
    Case("regex_remove_urls", remove_urls, {"text": STR}, "supported"),
    Case("regex_remove_punctuation", remove_punctuation, {"text": STR}, "supported"),
    Case("regex_keep_digits", keep_digits, {"text": STR}, "supported"),
    Case("regex_collapse_whitespace", collapse_whitespace, {"text": STR}, "supported"),
    Case("regex_first_number", first_number, {"text": STR}, "supported"),
]
