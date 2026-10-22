from collections.abc import Callable, Sequence
from random import getrandbits

from hypothesis import given, seed
from hypothesis import strategies as st

EXAMPLES_PER_QUERY = 5


def check_batched[Example](
    draw: Callable[[st.DataObject], Example], check: Callable[[Sequence[Example]], None]
) -> None:
    examples: list[Example] = []
    replay = False

    @seed(getrandbits(64))
    @given(st.data())
    def generate(data: st.DataObject) -> None:
        example = draw(data)
        if replay:
            check([example])
        else:
            examples.append(example)

    generate()
    try:
        for start in range(0, len(examples), EXAMPLES_PER_QUERY):
            check(examples[start : start + EXAMPLES_PER_QUERY])
    except Exception:
        replay = True
        generate()
        raise
