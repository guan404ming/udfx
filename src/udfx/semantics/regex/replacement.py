from __future__ import annotations

import re
import re._parser as parser

from udfx.errors import Unsupported


def replacement_has_newline(pattern: str, flags: int, value: str) -> bool:
    try:
        parts = parser.parse_template(value, re.compile(pattern, flags))
    except (re.error, IndexError):
        return True
    return any(isinstance(part, str) and "\n" in part for part in parts)


def replacement(pattern: str, flags: int, value: str, dialect: str) -> str:
    if any(0xD800 <= ord(point) <= 0xDFFF for point in value):
        raise Unsupported("regex replacement surrogates need fallback")
    try:
        parts = parser.parse_template(value, re.compile(pattern, flags))
    except (re.error, IndexError) as error:
        raise Unsupported(f"invalid regex replacement: {error}") from error
    result: list[str] = []
    for index, part in enumerate(parts):
        if isinstance(part, int):
            if dialect == "re2":
                if part > 9:
                    raise Unsupported("RE2 replacement supports groups 0 through 9")
                result.append(f"\\{part}")
            else:
                result.append(f"${part}" if dialect == "java" else f"${{{part}}}")
        elif dialect in {"rust", "datafusion"}:
            result.append(part.replace("$", "$$"))
        else:
            escaped_part = part.replace("\\", "\\\\")
            if (
                dialect == "java"
                and index
                and isinstance(parts[index - 1], int)
                and part
                and part[0] in "0123456789"
            ):
                escaped_part = "\\" + escaped_part
            result.append(escaped_part.replace("$", r"\$") if dialect == "java" else escaped_part)
    rendered = "".join(result)
    if dialect == "datafusion" and re.search(r"\\(?:[0-9]|[^\x00-\x7f])", rendered):
        raise Unsupported("DataFusion literal replacement backslashes before digits need fallback")
    return rendered
