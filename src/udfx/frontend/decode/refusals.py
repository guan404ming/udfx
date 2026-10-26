from __future__ import annotations

FEATURES: dict[str, str] = {
    "GET_ITER": "loops and comprehensions",
    "FOR_ITER": "loops and comprehensions",
    "JUMP_BACKWARD": "loops and comprehensions",
    "JUMP_BACKWARD_NO_INTERRUPT": "loops and comprehensions",
    "BUILD_LIST": "list and set literals",
    "BUILD_SET": "list and set literals",
    "LIST_APPEND": "list and set literals",
    "LIST_EXTEND": "list and set literals",
    "STORE_GLOBAL": "global assignment",
    "DELETE_GLOBAL": "global assignment",
    "BEFORE_WITH": "with blocks",
    "LOAD_SPECIAL": "with blocks",
    "WITH_EXCEPT_START": "with blocks",
    "RAISE_VARARGS": "raise statements",
    "YIELD_VALUE": "generators",
    "RETURN_GENERATOR": "generators",
    "CALL_FUNCTION_EX": "unpacked call arguments",
    "DICT_MERGE": "unpacked call arguments",
}
