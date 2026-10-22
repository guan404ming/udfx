import string

from udfx.types import STR

from . import Case


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def expand_letters(text: str) -> str:
    return text.translate(str.maketrans({"ß": "ss", "ä": "ae", "a": "b", "!": None}))


def swap_letters(text: str) -> str:
    return text.translate(str.maketrans("ab", "ba"))


CASES = [
    Case("translate_punctuation", remove_punctuation, {"text": STR.or_none()}, "native"),
    Case("translate_expansion", expand_letters, {"text": STR}, "native"),
    Case("translate_swap", swap_letters, {"text": STR}, "native"),
]
