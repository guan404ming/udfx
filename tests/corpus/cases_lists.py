from __future__ import annotations

from udfx.types import STR

from . import Case

STOPWORDS = frozenset({"a", "the", "of"})


def function_values(text: str) -> list[str]:
    return [word.split("=")[1] for word in text.split(";") if word.startswith("Func")]


def contains_word(text: str) -> bool:
    return any(word in text for word in ("a", "b"))


def unique_words(text: str) -> int:
    return len(set(text.split()))


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in STOPWORDS])


def total_word_length(text: str) -> int:
    return sum(len(word) for word in text.split())


def total_numbers(text: str) -> int:
    return sum(int(word) for word in text.split())


def all_words(text: str) -> bool:
    return all(word.startswith("a") for word in text.split())


def sorted_words(text: str) -> list[str]:
    return sorted(text.split())


def sorted_unique_words(text: str) -> list[str]:
    return sorted(set(text.split()))


def sliced_words(text: str) -> list[str]:
    return [word.upper() for word in text.split()[1:]]


def constant_suffixes(text: str) -> str:
    for suffix in ["!", "?"]:
        text += suffix
    return text


def selected_suffixes(text: str) -> str:
    result = ""
    for suffix in ("a", "b", "c"):
        if suffix in text:
            continue
        result += suffix
        if len(result) > len(text):
            break
    return result


def lazy_fault(text: str) -> bool:
    return any(word[0] in text for word in ("a", ""))


CASES = [
    Case(function.__name__, function, {"text": STR.or_none()}, "native")
    for function in (
        function_values,
        contains_word,
        unique_words,
        remove_stopwords,
        total_word_length,
        total_numbers,
        all_words,
        sorted_words,
        sorted_unique_words,
        sliced_words,
        constant_suffixes,
        selected_suffixes,
        lazy_fault,
    )
]
