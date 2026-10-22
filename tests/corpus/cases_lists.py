from __future__ import annotations

from udfx.types import INT, STR, list_of

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


def unpack_fields(text: str) -> str:
    key, value = text.split(",")
    return key + ":" + value


def unpack_words(text: str) -> str:
    first, *rest = text.split()
    return first + ":" + " ".join(rest)


def unpack_ends(text: str) -> str:
    first, *middle, last = text.split()
    return last + " ".join(middle) + first


def unpack_tuple(text: str) -> str:
    pair = (text, 2)
    value, stop = pair
    return value[:stop]


def unpack_constant(text: str) -> str:
    pair = ("prefix:", 2)
    prefix, stop = pair
    return prefix + text[:stop]


def unpack_caught(text: str) -> str:
    try:
        key, value = text.split("=")
        return key + value
    except ValueError:
        return text


def unpack_pair(pair: list[str]) -> str:
    key, value = pair
    return key + value


def unpack_numbers(numbers: list[int]) -> int:
    first, *middle, last = numbers
    return first + sum(middle) + last


def mapped_characters(text: str) -> list[int]:
    mapping = {"a": 1, "b": 2, " ": 0}
    return list(map(lambda character: mapping[character], list(text)))


def filtered_words(text: str) -> list[str]:
    return list(filter(lambda word: len(word) > 1, text.split()))


def numbered_words(text: str) -> list[str]:
    return [str(index) + word for index, word in enumerate(text.split(), 1)]


def paired_words(text: str) -> list[str]:
    return [left + right for left, right in zip(text.split(), ("a", "b"), strict=False)]


def word_chunks(text: str) -> list[list[str]]:
    return [text.split()[index : index + 6] for index in range(0, len(text.split()), 6)]


def unique_characters(text: str) -> int:
    return len(set(text))


def repeated_words(text: str) -> list[str]:
    return text.split() * 3


def dictionary_items(text: str) -> list[str]:
    return [key + str(value) for key, value in {"a": 1, "b": 2}.items() if key in text]


def weighted_word_length(text: str) -> int:
    result = 3
    for word, weight in zip(text.split(), (1, 2), strict=False):
        result += len(word) * weight
    return result


def indexed_words(text: str) -> str:
    result = "prefix:"
    for index, word in enumerate(text.split()):
        if word == "a":
            continue
        result += str(index) + word
    return result


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
        unpack_fields,
        unpack_words,
        unpack_ends,
        unpack_tuple,
        unpack_constant,
        unpack_caught,
        mapped_characters,
        filtered_words,
        numbered_words,
        paired_words,
        unique_characters,
        repeated_words,
        dictionary_items,
        weighted_word_length,
        indexed_words,
    )
]

CASES.extend(
    [
        Case("word_chunks", word_chunks, {"text": STR.or_none()}, "fallback"),
        Case("unpack_pair", unpack_pair, {"pair": list_of(STR).or_none()}, "native"),
        Case("unpack_numbers", unpack_numbers, {"numbers": list_of(INT).or_none()}, "native"),
    ]
)
