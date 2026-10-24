import json
import sys
from importlib.metadata import version
from pathlib import Path

import udfx
from udfx.backends import named
from udfx.types import FLOAT, INT, STR


def prefix(text: str) -> str:
    return text[:3]


def float_text(value: float) -> str:
    return str(value)


def word_lengths(text: str) -> list[int]:
    return [len(word) for word in text.split()]


def inverse(value: int) -> float:
    return 1 / value


def main() -> None:
    name = sys.argv[1]
    package = Path(udfx.__file__).resolve()
    assert not package.is_relative_to(Path(__file__).resolve().parents[2]), package
    assert package.with_name("py.typed").is_file()
    backend = named(name)
    for function, schema, rows in (
        (prefix, {"text": STR}, [("hello",), ("İß你好",), ("",)]),
        (float_text, {"value": FLOAT}, [(1.0000000000000002,), (-0.0,), (1e-30,)]),
        (word_lengths, {"text": STR}, [("one two",), ("你好\t世界",), ("",)]),
    ):
        translation = udfx.compile(function, schema, backend)
        assert translation.ok, translation.unsupported
        actual = backend.evaluate([translation.expr], schema, rows)[0]
        assert actual == [function(*row) for row in rows], actual
        report = udfx.explain(function, schema, backend)
        assert f"Backend: {name}" in report, report
        if function is float_text:
            assert "Python fallback: possible" in report, report
    translation = udfx.compile(inverse, {"value": INT}, backend)
    assert translation.ok, translation.unsupported
    try:
        backend.evaluate([translation.expr], {"value": INT}, [(0,)])
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("Python exception was not restored")
    engine = "pyspark" if name == "spark" else name
    print(json.dumps({"udfx": udfx.__version__, "engine": engine, "version": version(engine)}))


if __name__ == "__main__":
    main()
