from itertools import product

from udfx.ops.registry import Op, register
from udfx.semantics.randomness import gauss
from udfx.types import BOOL, FLOAT, INT

for count in range(3):
    for parameters in product((BOOL, INT, FLOAT), repeat=count):
        register(Op("random.gauss", parameters, FLOAT, gauss, deterministic=False))
