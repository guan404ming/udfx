from datafusion import Expr, functions, lit

from udfx.backends.datafusion.backend import DataFusionBackend
from udfx.backends.datafusion.conversions import TEXT
from udfx.backends.datafusion.rules import RULES
from udfx.backends.registration_json import register_json_string_decoder, register_json_values
from udfx.frontend.core.json import serialized_ascii_lists


def decode(value: Expr) -> Expr:
    document = functions.concat(lit('{"value":'), value, lit("}"))
    return functions.spark.json_tuple(document, "value")["c0"]


register_json_string_decoder(RULES, TEXT, decode)
register_json_values(
    RULES, DataFusionBackend().literal, TEXT, serialize_strings=serialized_ascii_lists
)
