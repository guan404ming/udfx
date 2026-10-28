from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrequencyPrimitives:
    literal: Callable[[object], Any]
    entries: Callable[[Any], Any]
    entry_key: Callable[[Any], Any]
    transform: Callable[[Any, Callable[[Any], Any]], Any]
    count: Callable[[Any, Any], Any]
    position: Callable[[Any, Any], Any]
    record: Callable[[Any, Any, Any], Any]
    sort: Callable[[Any], Any]
    key: Callable[[Any], Any]


def ordered_frequencies(primitives: FrequencyPrimitives, values: Any) -> Any:
    ranked = primitives.transform(
        primitives.entries(values),
        lambda entry: primitives.record(
            primitives.literal(0) - primitives.count(values, entry),
            primitives.position(values, primitives.entry_key(entry)),
            primitives.entry_key(entry),
        ),
    )
    return primitives.transform(primitives.sort(ranked), primitives.key)


@dataclass(frozen=True)
class FrequencySequencePrimitives:
    literal: Callable[[object], Any]
    length: Callable[[Any], Any]
    indices: Callable[[Any], Any]
    field: Callable[[Any, int], Any]
    record: Callable[..., Any]
    sort: Callable[[Any], Any]
    transform: Callable[[Any, Callable[[Any], Any]], Any]
    filter: Callable[[Any, Callable[[Any], Any]], Any]
    bind: Callable[[Any, Callable[[Any], Any]], Any]


@dataclass(frozen=True)
class FrequencyRunPrimitives(FrequencySequencePrimitives):
    zip: Callable[..., Any]
    head: Callable[[Any], Any]
    tail: Callable[[Any], Any]
    concatenate: Callable[[Any, Any], Any]
    append: Callable[[Any, Any], Any]


@dataclass(frozen=True)
class FrequencyOffsetPrimitives(FrequencySequencePrimitives):
    get: Callable[[Any, Any], Any]
    coalesce: Callable[[Any, Any], Any]
    position: Callable[[Any, Any], Any]
    maximum: Callable[[Any], Any]
    prefix: Callable[[Any, Any], Any]
    distinct: Callable[[Any], Any]
    indexed: Callable[[Any, Callable[[Any, Any], Any]], Any]
    logarithm: Callable[[Any], Any]
    select: Callable[[Any, Any, Any], Any]


def indexed_values(primitives: FrequencyRunPrimitives, values: Any) -> Any:
    return primitives.sort(primitives.zip(values, primitives.indices(primitives.length(values))))


def sorted_frequencies(primitives: FrequencyRunPrimitives, values: Any) -> Any:
    indexed = indexed_values(primitives, values)
    return primitives.bind(indexed, lambda items: frequency_runs(primitives, items, values))


def offset_frequencies(primitives: FrequencyOffsetPrimitives, values: Any, count: Any) -> Any:
    return primitives.select(
        count <= primitives.literal(0),
        primitives.prefix(values, primitives.literal(0)),
        primitives.bind(
            primitives.sort(values),
            lambda items: frequency_offsets(primitives, items, values, count),
        ),
    )


def frequency_offsets(
    primitives: FrequencyOffsetPrimitives, items: Any, values: Any, count: Any
) -> Any:
    starts = primitives.filter(
        primitives.indices(primitives.length(items)),
        lambda index: (
            (index == primitives.literal(0))
            | (primitives.get(items, index) != primitives.get(items, index - primitives.literal(1)))
        ),
    )

    def ranked(boundaries: Any) -> Any:
        counts = primitives.transform(
            primitives.indices(primitives.length(boundaries)),
            lambda index: (
                primitives.get(boundaries, index)
                - primitives.coalesce(
                    primitives.get(boundaries, index + primitives.literal(1)),
                    primitives.length(items),
                )
            ),
        )
        return primitives.bind(
            counts,
            lambda sizes: rank_frequency_candidates(
                primitives, values, items, boundaries, sizes, count=count
            ),
        )

    return primitives.bind(starts, ranked)


def rank_frequency_candidates(
    primitives: FrequencyOffsetPrimitives,
    values: Any,
    items: Any,
    boundaries: Any,
    sizes: Any,
    *,
    count: Any,
) -> Any:
    cutoff = primitives.coalesce(
        primitives.maximum(primitives.prefix(primitives.sort(sizes), count)), primitives.literal(0)
    )

    def ranked(threshold: Any) -> Any:
        selected = primitives.filter(
            primitives.indices(primitives.length(sizes)),
            lambda index: primitives.get(sizes, index) <= threshold,
        )

        def ordered(candidates: Any, positions: Any = None) -> Any:
            def entry(index: Any) -> Any:
                key = primitives.get(items, primitives.get(boundaries, index))
                position = (
                    primitives.position(values, key)
                    if positions is None
                    else primitives.field(primitives.get(positions, index), 1)
                )
                return primitives.record(primitives.get(sizes, index), position, key)

            records = primitives.sort(primitives.transform(candidates, entry))
            return primitives.transform(
                primitives.prefix(records, count), lambda item: primitives.field(item, 2)
            )

        def choose(candidates: Any) -> Any:
            size = primitives.length(values)
            limit = primitives.logarithm(
                primitives.select(size > primitives.literal(0), size, primitives.literal(1))
            )
            positions = primitives.sort(
                primitives.indexed(primitives.distinct(values), primitives.record)
            )
            return primitives.select(
                primitives.length(candidates) <= limit,
                ordered(candidates),
                primitives.bind(positions, lambda order: ordered(candidates, order)),
            )

        return primitives.bind(selected, choose)

    return primitives.bind(cutoff, ranked)


def frequency_runs(primitives: FrequencyRunPrimitives, indexed: Any, values: Any) -> Any:
    neighbors = primitives.zip(
        indexed,
        primitives.concatenate(primitives.head(indexed), indexed),
        primitives.indices(primitives.length(indexed)),
    )
    starts = primitives.filter(
        neighbors,
        lambda item: (
            (primitives.field(item, 2) == primitives.literal(0))
            | (
                primitives.field(primitives.field(item, 0), 0)
                != primitives.field(primitives.field(item, 1), 0)
            )
        ),
    )
    boundaries = primitives.transform(
        starts,
        lambda item: primitives.record(
            primitives.field(primitives.field(item, 0), 0),
            primitives.field(primitives.field(item, 0), 1),
            primitives.field(item, 2),
        ),
    )
    return primitives.bind(boundaries, lambda items: rank_frequency_runs(primitives, items, values))


def rank_frequency_runs(primitives: FrequencyRunPrimitives, boundaries: Any, values: Any) -> Any:
    offsets = primitives.transform(boundaries, lambda item: primitives.field(item, 2))
    ends = primitives.tail(primitives.append(offsets, primitives.length(values)))
    ranked = primitives.transform(
        primitives.zip(boundaries, ends),
        lambda item: primitives.record(
            primitives.field(primitives.field(item, 0), 2) - primitives.field(item, 1),
            primitives.field(primitives.field(item, 0), 1),
            primitives.field(primitives.field(item, 0), 0),
        ),
    )
    return primitives.transform(primitives.sort(ranked), lambda item: primitives.field(item, 2))
