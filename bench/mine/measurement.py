from __future__ import annotations

import ast
import re

import udfx
from bench.mine.bindings import bind_candidate
from bench.mine.models import Candidate, Result
from bench.mine.schema_search import schemas


def normalize_reason(reason: str) -> str:
    reason = re.sub(r" at [\w.<>]+\+\d+$", "", reason)
    return re.sub(r"0x[0-9a-fA-F]+", "<address>", reason)


def failure_category(reason: str) -> str:
    reason = normalize_reason(reason)
    for prefix in ("unknown callable ", "unknown global "):
        if reason.startswith(prefix):
            return prefix.rstrip()
    if reason.startswith("no op "):
        return reason.split("(", 1)[0]
    return reason


def measure(candidate: Candidate) -> Result:
    result = Result(
        candidate.kernel_id, candidate.occurrences, ast.unparse(candidate.node), kind=candidate.kind
    )
    try:
        function = bind_candidate(candidate)
    except ModuleNotFoundError as error:
        result.reason = result.category = str(error)
        return result
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        result.reason = f"definition error: {type(error).__name__}: {error}"
        result.category = f"definition error: {type(error).__name__}"
        return result
    try:
        attempts = list(schemas(candidate))
    except ValueError as error:
        result.reason = result.category = str(error)
        return result
    for schema in attempts:
        try:
            translation = udfx.compile(function, schema)
        except Exception as error:
            reason = f"compiler error: {type(error).__name__}: {error}"
        else:
            if translation.ok:
                result.ok = True
                result.schema = {name: str(value) for name, value in schema.items()}
                result.reason = None
                result.category = None
                return result
            assert translation.unsupported is not None
            reason = str(translation.unsupported)
        if candidate.kind == "row" and reason.startswith("schema "):
            result.reason = f"row schema unsupported: {reason}"
            result.category = "row schema unsupported"
            return result
        if result.reason is None:
            result.reason = reason
            result.category = failure_category(reason)
    return result
