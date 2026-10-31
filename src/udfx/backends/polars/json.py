import polars as pl

from udfx.backends.polars.backend import PolarsBackend
from udfx.backends.polars.conversions import TEXT
from udfx.backends.polars.rules import RULES
from udfx.backends.registration_json import register_json_string_decoder, register_json_values
from udfx.backends.rules import Lowered
from udfx.frontend.core.json import nonempty_array, serialized_words
from udfx.semantics.facts import json_validity
from udfx.semantics.json import lower_decoded_strings
from udfx.types import Kind

register_json_string_decoder(RULES, TEXT, lambda value: value.str.json_decode(pl.String))

RULES("json.result.list[str]", Kind.JSON, proves=json_validity)(
    lambda value: Lowered(
        *lower_decoded_strings(
            TEXT,
            value,
            lambda text, exact: (
                pl.when(exact)
                .then(text)
                .otherwise(pl.lit("[]"))
                .str.json_decode(pl.List(pl.String))
            ),
        )
    )
)

register_json_values(RULES, PolarsBackend().literal, TEXT, serialize_strings=serialized_words)
RULES.rewrite("__gt__", Kind.INT, Kind.INT)(nonempty_array)
