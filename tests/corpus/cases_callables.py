import pytest

from udfx.types import FLOAT, INT, STR

from . import Case

numpy = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")


def pandas_missing(value: object):
    return pandas.isna(value)


def pandas_null(value: object):
    return pandas.isnull(value)


def pandas_present(value: object):
    return pandas.notna(value)


def pandas_not_null(value: object):
    return pandas.notnull(value)


def numpy_missing(value: object):
    return numpy.isnan(value)


def numpy_log(value: object):
    return numpy.log(value)


def numpy_root(value: object):
    return numpy.sqrt(value)


def numpy_exponential(value: object):
    return numpy.exp(value)


def numpy_absolute(value: object):
    return numpy.abs(value)


def numpy_round(value: object):
    return numpy.round(value)


def numpy_round_digits(value: float, digits: int):
    return numpy.round(value, digits)


def numpy_floor(value: object):
    return numpy.floor(value)


def numpy_ceil(value: object):
    return numpy.ceil(value)


CASES = [
    Case("pandas_missing", pandas_missing, {"value": FLOAT.or_none()}, "native"),
    Case("pandas_null", pandas_null, {"value": STR.or_none()}, "native"),
    Case("pandas_present", pandas_present, {"value": INT.or_none()}, "native"),
    Case("pandas_not_null", pandas_not_null, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_missing", numpy_missing, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_log", numpy_log, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_integer_log", numpy_log, {"value": INT}, "native"),
    Case("numpy_root", numpy_root, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_integer_root", numpy_root, {"value": INT}, "native"),
    Case("numpy_exponential", numpy_exponential, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_integer_exponential", numpy_exponential, {"value": INT}, "native"),
    Case("numpy_absolute", numpy_absolute, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_round", numpy_round, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_round_digits", numpy_round_digits, {"value": FLOAT, "digits": INT}, "native"),
    Case("numpy_floor", numpy_floor, {"value": FLOAT.or_none()}, "native"),
    Case("numpy_ceil", numpy_ceil, {"value": FLOAT.or_none()}, "native"),
]
