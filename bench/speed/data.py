import inspect
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from types import CodeType

import pyarrow

from tests.corpus.cases import Case
from udfx.backends import arrow
from udfx.frontend.opcodes import normalize
from udfx.program import Program
from udfx.testing.outcome import Returned, run
from udfx.types import Kind, Type

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
        instructions = normalize(code)
        for receiver, access, operation in zip(
            instructions, instructions[1:], instructions[2:], strict=False
        ):
            if receiver.opname not in {"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_DEREF"}:
                continue
            if receiver.argval != name:
                continue
            if access.opname == "LOAD_ATTR" and access.argval in program.schema:
                attributes = True
            elif access.opname == "LOAD_CONST" and operation.opname == "BINARY_SUBSCR":
                positional |= isinstance(access.argval, int)
                named |= isinstance(access.argval, str)
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
    return pyarrow.table(columns), expected
