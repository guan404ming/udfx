import inspect
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from types import CodeType

import pyarrow

from tests.corpus.cases import Case
from udfx.backends import arrow
from udfx.frontend.decode import decode
from udfx.frontend.decode.ops import Attribute, Constant, Local, Subscript
from udfx.program import Program
from udfx.testing.outcome import Returned, run
from udfx.types import Kind, Type

BATCH_SIZE = 8192

SAMPLE_DATES = ["2024-05-17", "1999-12-31", "", "20x4-01-01", " 123-1"]
SAMPLE_WORDS = ["Hello World", "funder::class::id", "a::b", "", "  padded  ", "42", " -3 "]
EXTRA_WORDS = [
    "key,value",
    "x=1",
    "Func=a;x;Func=b",
    "see https://example.com now",
    "2024-05-17 10:00",
    "a",
    "c",
]
MIXED_CASES = {"regex_remove_urls", "function_values", "dictionary_none", "unpack_ends"}
EPOCH = datetime(1970, 1, 1)


def generate(name: str, type: Type, seed: random.Random, words: list[str]) -> object:
    if type.nullable and seed.random() < 0.1:
        return None
    pool = SAMPLE_DATES if "date" in name else words
    match type.kind:
        case Kind.STR:
            return seed.choice(pool)
        case Kind.LIST:
            assert type.item is not None
            return [generate(name, type.item, seed, words) for _ in range(seed.randint(0, 3))]
        case Kind.INT if name == "index":
            return seed.randint(-3, 3)
        case Kind.INT if name == "width":
            return seed.randint(-5, 30)
        case Kind.INT:
            return seed.randint(-1000, 1000)
        case Kind.FLOAT:
            return seed.uniform(-1000.0, 1000.0)
        case Kind.BOOL:
            return seed.random() < 0.5
        case Kind.DATETIME:
            return EPOCH + timedelta(microseconds=seed.randint(0, 2_000_000_000_000_000))
        case _:
            raise ValueError(f"no generator for {type}")


def with_arity[Function: Callable[..., object]](function: Function, count: int) -> Function:
    parameters = [
        inspect.Parameter(f"value{index}", inspect.Parameter.POSITIONAL_ONLY)
        for index in range(count)
    ]
    function.__signature__ = inspect.Signature(parameters)
    return function


class AttributeRow(dict[str, object]):
    __getattr__ = dict.__getitem__


class MixedRow(AttributeRow):
    def __getitem__(self, key: str | int) -> object:
        if isinstance(key, int):
            return tuple(dict.values(self))[key]
        return dict.__getitem__(self, key)


def row_type(program: Program) -> type[dict[str, object]] | type[tuple[object, ...]]:
    codes = [program.function.__code__]
    name = codes[0].co_varnames[0]
    positional = named = attributes = False
    while codes:
        code = codes.pop()
        codes.extend(
            value
            for value in code.co_consts
            if isinstance(value, CodeType) and name in value.co_freevars
        )
        operations = decode(code).operations(0)
        constants = {op.result: op.value for op in operations if isinstance(op, Constant)}
        receivers = {op.result for op in operations if isinstance(op, Local) and op.name == name}
        for op in operations:
            if isinstance(op, Attribute) and op.owner in receivers and op.name in program.schema:
                attributes = True
            elif isinstance(op, Subscript) and op.container in receivers:
                key = constants.get(op.index)
                positional |= isinstance(key, int)
                named |= isinstance(key, str)
    if not (positional or named or attributes):
        return MixedRow
    if positional:
        return MixedRow if named or attributes else tuple
    return AttributeRow if attributes else dict


def row_function(program: Program) -> Callable[..., object]:
    function = program.function
    constructor = row_type(program)
    if constructor in (dict, tuple):
        return function
    return lambda row: function(constructor(row))


def caller(program: Program) -> Callable[..., object]:
    if not program.row_mode:
        return program.function
    function = program.function
    constructor = row_type(program)
    names = tuple(program.schema)
    if constructor is tuple:
        return with_arity(lambda *values: function(values), len(names))
    return with_arity(
        lambda *values: function(constructor(zip(names, values, strict=True))), len(names)
    )


def batched(
    function: Callable[..., object], result: pyarrow.DataType, count: int
) -> Callable[..., pyarrow.Array]:
    def apply(*arrays: pyarrow.Array) -> pyarrow.Array:
        columns = [array.to_pylist() for array in arrays]
        return pyarrow.array([function(*row) for row in zip(*columns, strict=True)], result)

    return with_arity(apply, count)


def sampled(
    program: Program, row_count: int, words: list[str]
) -> tuple[list[tuple[object, ...]], list[object]]:
    seed = random.Random(0)
    function = caller(program)
    rows: list[tuple[object, ...]] = []
    expected: list[object] = []
    for _ in range(row_count * 20):
        if len(rows) == row_count:
            break
        row = tuple(generate(name, type, seed, words) for name, type in program.schema.items())
        outcome = run(function, row, program.result)
        if isinstance(outcome, Returned):
            rows.append(row)
            expected.append(outcome.value)
    return rows, expected


def dataset(case: Case, program: Program, row_count: int) -> tuple[pyarrow.Table, list[object]]:
    rows, expected = sampled(program, row_count, SAMPLE_WORDS)
    if len(rows) < row_count or case.name in MIXED_CASES:
        rows, expected = sampled(program, row_count, SAMPLE_WORDS + EXTRA_WORDS)
    if len(rows) < row_count:
        raise ValueError(f"only {len(rows)} generated rows run without an exception")
    columns = {
        name: pyarrow.array([row[index] for row in rows], arrow.dtype(type))
        for index, (name, type) in enumerate(program.schema.items())
    }
    table = pyarrow.table(columns)
    return pyarrow.Table.from_batches(table.to_batches(BATCH_SIZE), table.schema), expected
