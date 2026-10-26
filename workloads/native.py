import datafusion
import duckdb
import polars
from datafusion import functions as fusion_functions
from pyspark.sql import functions as spark_functions

YEAR_MONTH = {
    "polars": lambda: polars.col("date").str.slice(0, 7),
    "duckdb": lambda: duckdb.FunctionExpression(
        "substring",
        duckdb.ColumnExpression("date"),
        duckdb.ConstantExpression(1),
        duckdb.ConstantExpression(7),
    ),
    "datafusion": lambda: fusion_functions.substr(datafusion.col("date"), 1, 7),
    "spark": lambda: spark_functions.substring("date", 1, 7),
}
