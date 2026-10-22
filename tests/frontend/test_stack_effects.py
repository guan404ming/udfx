import dis
import sys

import pytest

from tests.frontend.stack_effect_tables import STACK_EFFECTS
from udfx.frontend.decode.refusals import FEATURES
from udfx.frontend.decode.tables import OVERRIDES, resolve

VERSION = sys.version_info[:2]


def stack_effect(name: str, jump: bool) -> int:
    code = dis.opmap[name]
    return dis.stack_effect(code, 2 if code >= dis.HAVE_ARGUMENT else None, jump=jump)


@pytest.mark.parametrize("name", sorted(STACK_EFFECTS[VERSION]))
def test_named_opcode_keeps_its_stack_effect(name: str) -> None:
    assert (stack_effect(name, False), stack_effect(name, True)) == STACK_EFFECTS[VERSION][name]


def test_decoder_names_only_opcodes_with_a_pinned_stack_effect() -> None:
    named = resolve(VERSION).keys() & dis.opmap.keys()
    assert named <= STACK_EFFECTS[VERSION].keys()


def test_every_pinned_opcode_is_decoded_or_refused() -> None:
    assert STACK_EFFECTS[VERSION].keys() <= resolve(VERSION).keys() | FEATURES.keys()


def test_overrides_name_only_real_opcodes_of_some_version() -> None:
    named = {name for overrides in OVERRIDES.values() for name in overrides}
    pinned = {name for effects in STACK_EFFECTS.values() for name in effects}
    assert named <= pinned
