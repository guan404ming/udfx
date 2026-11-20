from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Mapping, Sequence
from types import CodeType, FunctionType, ModuleType
from typing import ClassVar

from pyspark.sql import Column, SparkSession, functions
from pyspark.sql.types import BooleanType, DataType, DoubleType, LongType, StringType

from udfx.api import compile
from udfx.backends.diagnostics import capture_lowering
from udfx.backends.engines.spark.backend import SparkBackend
from udfx.backends.engines.spark.dtypes import dtype
from udfx.backends.engines.spark.rules import ANSI_ARITHMETIC
from udfx.backends.lowering.programs import lower_program
from udfx.backends.lowering.results import ResultLowering
from udfx.backends.registration.shared import Lowered
from udfx.ir import Col
from udfx.program import Program
from udfx.types import BOOL, FLOAT, INT, STR, Kind, Type

MODULE = "pyspark.sql.transpile"
ENABLED = "spark.sql.experimental.optimizer.transpilePyUDFs"
TRANSPILERS = "spark.sql.experimental.optimizer.pyTranspilers"
ANSI = "spark.sql.ansi.enabled"
VARIETY = "udfx"
CATEGORIES: dict[str, Type] = {"integer": INT, "float": FLOAT, "string": STR, "bool": BOOL}
RESULTS: dict[Kind, type[DataType]] = {
    Kind.INT: LongType,
    Kind.FLOAT: DoubleType,
    Kind.STR: StringType,
    Kind.BOOL: BooleanType,
}


def transpile_module() -> ModuleType | None:
    return importlib.import_module(MODULE) if importlib.util.find_spec(MODULE) else None


class ParameterBackend(SparkBackend):
    graphs: ClassVar[bool] = False
    hoisted: ClassVar[frozenset[Kind]] = frozenset()

    def __init__(self, session: SparkSession, positions: Mapping[str, int]) -> None:
        super().__init__(session)
        self.positions = positions
        self.raised: Column | None = None

    def column(self, node: Col) -> Column:
        return functions.col(f"_udf_param_{self.positions[node.name]}").cast(dtype(node.type))

    def lower_program(self, program: Program) -> Lowered[Column]:
        token = ANSI_ARITHMETIC.set(self.session().conf.get(ANSI) == "true")
        try:
            return lower_program(self, program, ResultLowering(self, program))
        finally:
            ANSI_ARITHMETIC.reset(token)

    def native_raise(self, program: Program) -> Column:
        name = getattr(program.function, "__qualname__", "function")
        message = functions.lit(f"Python raises an exception in {name} for this row (udfx)")
        self.raised = functions.raise_error(message).cast(dtype(program.result))
        return self.raised

    def select(self, condition: Column, then: Column, orelse: Column) -> Column:
        if then is self.raised:
            return functions.least(orelse, functions.when(condition, then))
        return super().select(condition, then, orelse)


def caller(code: CodeType) -> Mapping[str, object] | None:
    frame = inspect.currentframe()
    while frame is not None and frame.f_code is not code:
        frame = frame.f_back
    return None if frame is None else frame.f_locals


def matches(type: Type, result: DataType) -> bool:
    expected = RESULTS.get(type.kind)
    return expected is not None and isinstance(result, expected)


def lower(
    function: object,
    parameters: Sequence[str],
    categories: Mapping[int, str],
    result: DataType,
    session: SparkSession,
) -> Column | None:
    if not isinstance(function, FunctionType):
        return None
    code = function.__code__
    if list(code.co_varnames[: code.co_argcount]) != list(parameters):
        return None
    schema: dict[str, Type] = {}
    for index, name in enumerate(parameters):
        type = CATEGORIES.get(categories.get(index, ""))
        if type is None:
            return None
        schema[name] = type.or_none()
    backend = ParameterBackend(session, {name: index for index, name in enumerate(parameters)})
    with capture_lowering() as details:
        translation = compile(function, schema, backend)
    program = translation.program
    if (
        translation.expr is None
        or program is None
        or not program.deterministic
        or not details
        or details[-1].exact is not None
        or not matches(program.result, result)
        or " -> " in str(translation.expr)
    ):
        return None
    return translation.expr.cast(result)


class Transpiler:
    variety = VARIETY

    def _transpile_from_ast(  # noqa: PLR0917
        self,
        source: str | None,
        tree: object,
        definition: object,
        parameters: list[str],
        result: DataType,
        categories: Mapping[int, str] | None = None,
    ) -> Column | None:
        module = transpile_module()
        call = None if module is None else caller(module._transpile_func.__code__)  # pyright: ignore[reportPrivateUsage]
        if call is None or not isinstance(session := call.get("session"), SparkSession):
            return None
        return lower(call.get("func"), parameters, categories or {}, result, session)


def install(session: SparkSession | None = None) -> None:
    module = transpile_module()
    if module is None:
        raise RuntimeError(f"{MODULE} is missing; install a pyspark with the Python UDF transpiler")
    module.AbstractTranspiler.varieties[VARIETY] = Transpiler
    active = session or SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    configured = (active.conf.get(TRANSPILERS) or "").split(",")
    names = [VARIETY, *(name for name in configured if name not in ("", VARIETY))]
    active.conf.set(TRANSPILERS, ",".join(names))
    active.conf.set(ENABLED, "true")
