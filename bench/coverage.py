import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from corpus.cases import CASES
from corpus.udfbench import load

from udfx import compile


def main() -> None:
    cases = [*CASES, *(load() or [])]
    translated = 0
    print(f"{'name':<28}{'translated':<12}reason")
    for case in cases:
        translation = compile(case.function, case.schema)
        translated += translation.ok
        reason = "" if translation.ok else str(translation.unsupported)
        print(f"{case.name:<28}{'yes' if translation.ok else 'no':<12}{reason}")
    print(f"translated {translated}/{len(cases)} ({100 * translated / len(cases):.1f}%)")


if __name__ == "__main__":
    main()
