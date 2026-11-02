from pathlib import Path

import pyarrow
from pyarrow import compute, csv

INTEGER = pyarrow.int64()
NUMBER = pyarrow.float64()
TEXT = pyarrow.string()
DATE = pyarrow.date32()

SCHEMAS = {
    "lineitem": pyarrow.schema(
        [
            ("l_orderkey", INTEGER),
            ("l_partkey", INTEGER),
            ("l_suppkey", INTEGER),
            ("l_linenumber", INTEGER),
            ("l_quantity", NUMBER),
            ("l_extendedprice", NUMBER),
            ("l_discount", NUMBER),
            ("l_tax", NUMBER),
            ("l_returnflag", TEXT),
            ("l_linestatus", TEXT),
            ("l_shipdate", DATE),
            ("l_commitdate", DATE),
            ("l_receiptdate", DATE),
            ("l_shipinstruct", TEXT),
            ("l_shipmode", TEXT),
            ("l_comment", TEXT),
        ]
    ),
    "orders": pyarrow.schema(
        [
            ("o_orderkey", INTEGER),
            ("o_custkey", INTEGER),
            ("o_orderstatus", TEXT),
            ("o_totalprice", NUMBER),
            ("o_orderdate", DATE),
            ("o_orderpriority", TEXT),
            ("o_clerk", TEXT),
            ("o_shippriority", INTEGER),
            ("o_comment", TEXT),
        ]
    ),
    "part": pyarrow.schema(
        [
            ("p_partkey", INTEGER),
            ("p_name", TEXT),
            ("p_mfgr", TEXT),
            ("p_brand", TEXT),
            ("p_type", TEXT),
            ("p_size", INTEGER),
            ("p_container", TEXT),
            ("p_retailprice", NUMBER),
            ("p_comment", TEXT),
        ]
    ),
    "partsupp": pyarrow.schema(
        [
            ("ps_partkey", INTEGER),
            ("ps_suppkey", INTEGER),
            ("ps_availqty", INTEGER),
            ("ps_supplycost", NUMBER),
            ("ps_comment", TEXT),
        ]
    ),
    "supplier": pyarrow.schema(
        [
            ("s_suppkey", INTEGER),
            ("s_name", TEXT),
            ("s_address", TEXT),
            ("s_nationkey", INTEGER),
            ("s_phone", TEXT),
            ("s_acctbal", NUMBER),
            ("s_comment", TEXT),
        ]
    ),
    "customer": pyarrow.schema(
        [
            ("c_custkey", INTEGER),
            ("c_name", TEXT),
            ("c_address", TEXT),
            ("c_nationkey", INTEGER),
            ("c_phone", TEXT),
            ("c_acctbal", NUMBER),
            ("c_mktsegment", TEXT),
            ("c_comment", TEXT),
        ]
    ),
    "nation": pyarrow.schema(
        [("n_nationkey", INTEGER), ("n_name", TEXT), ("n_regionkey", INTEGER), ("n_comment", TEXT)]
    ),
    "region": pyarrow.schema([("r_regionkey", INTEGER), ("r_name", TEXT), ("r_comment", TEXT)]),
}


def read(path: Path) -> pyarrow.Table:
    if path.suffix != ".tbl" or path.stem not in SCHEMAS:
        raise ValueError(f"Unknown TPC-H table: {path.name}")
    schema = SCHEMAS[path.stem]
    ending = "tpch trailing field"
    table = csv.read_csv(
        path,
        read_options=csv.ReadOptions(column_names=[*schema.names, ending]),
        parse_options=csv.ParseOptions(delimiter="|", quote_char=False),
        convert_options=csv.ConvertOptions(
            column_types={**{field.name: field.type for field in schema}, ending: TEXT},
            null_values=[],
            strings_can_be_null=False,
        ),
    )
    if compute.max(compute.utf8_length(table[ending])).as_py():
        raise ValueError(f"TPC-H rows must end with an empty field: {path}")
    return table.select(schema.names)


def load(paths: list[Path]) -> dict[str, pyarrow.Table]:
    tables = {}
    for path in paths:
        if path.stem in tables:
            raise ValueError(f"Duplicate TPC-H table: {path.stem}")
        tables[path.stem] = read(path)
    if not tables:
        raise ValueError("Supply local TPC-H .tbl inputs")
    return tables


def join(
    left: pyarrow.Table, right: pyarrow.Table, key: str | list[str], foreign_key: str | list[str]
) -> pyarrow.Table:
    left_position, right_position = "qure left position", "qure right position"
    left = left.append_column(left_position, pyarrow.array(range(left.num_rows), INTEGER))
    right = right.append_column(right_position, pyarrow.array(range(right.num_rows), INTEGER))
    return (
        left.join(
            right,
            keys=key,
            right_keys=foreign_key,
            join_type="inner",
            coalesce_keys=False,
            use_threads=False,
        )
        .sort_by([(left_position, "ascending"), (right_position, "ascending")])
        .drop([left_position, right_position])
    )
