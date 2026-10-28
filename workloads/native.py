import importlib


def polars_year_month():
    return importlib.import_module("polars").col("date").str.slice(0, 7)


def duckdb_year_month():
    engine = importlib.import_module("duckdb")
    return engine.FunctionExpression(
        "substring",
        engine.ColumnExpression("date"),
        engine.ConstantExpression(1),
        engine.ConstantExpression(7),
    )


def datafusion_year_month():
    engine = importlib.import_module("datafusion")
    return engine.functions.substr(engine.col("date"), 1, 7)


def spark_year_month():
    return importlib.import_module("pyspark.sql.functions").substring("date", 1, 7)


YEAR_MONTH = {
    "polars": polars_year_month,
    "duckdb": duckdb_year_month,
    "datafusion": datafusion_year_month,
    "spark": spark_year_month,
}
