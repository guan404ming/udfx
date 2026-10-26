# pyright: basic
from __future__ import annotations

from udfx.types import INT, STR

from . import Case


def split_limit(text, separator, maximum):
    return text.split(separator, maximum)


def rsplit_limit(text, separator, maximum):
    return text.rsplit(separator, maximum)


def split_word_limit(text, maximum):
    return text.split(None, maximum)


def rsplit_word_limit(text, maximum):
    return text.rsplit(None, maximum)


def rsplit_words(text):
    return text.rsplit()


def rsplit_by(text, separator):
    return text.rsplit(separator)


def index_split_limit(text, maximum, index):
    try:
        return text.split("_", maximum)[index]
    except IndexError:
        return None


def index_rsplit_limit(text, maximum, index):
    try:
        return text.rsplit("_", maximum)[index]
    except IndexError:
        return None


CASES = [
    Case(
        "split_limit",
        split_limit,
        {"text": STR, "separator": STR.or_none(), "maximum": INT},
        "supported",
    ),
    Case(
        "rsplit_limit",
        rsplit_limit,
        {"text": STR, "separator": STR.or_none(), "maximum": INT},
        "supported",
    ),
    Case("split_word_limit", split_word_limit, {"text": STR, "maximum": INT}, "supported"),
    Case("rsplit_word_limit", rsplit_word_limit, {"text": STR, "maximum": INT}, "supported"),
    Case("rsplit_words", rsplit_words, {"text": STR}, "supported"),
    Case("rsplit_by", rsplit_by, {"text": STR, "separator": STR.or_none()}, "supported"),
    Case(
        "index_split_limit",
        index_split_limit,
        {"text": STR, "maximum": INT, "index": INT},
        "supported",
    ),
    Case(
        "index_rsplit_limit",
        index_rsplit_limit,
        {"text": STR, "maximum": INT, "index": INT},
        "supported",
    ),
]
