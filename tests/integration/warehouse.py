"""Local Iceberg warehouse for integration tests.

Creates two tables in the ``demo`` namespace that mirror the canary sample data
(tests/canary/sample_data/exports.yaml):

  demo.dim_listings      — partitioned by ``ds`` (date string)
  demo.user_activities   — partitioned by ``ds`` (date string), event_time is TIMESTAMP

Both ``demo`` and ``data`` namespaces are created so the Scala batch driver can write
output tables into ``data.*``.
"""

from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

DIM_LISTINGS_SCHEMA = StructType(
    [
        StructField("listing_id", LongType(), True),
        StructField("merchant_id", LongType(), True),
        StructField("created_at_ts", LongType(), True),  # epoch ms
        StructField("updated_at_ts", LongType(), True),  # epoch ms
        StructField("is_active", BooleanType(), True),
        StructField("headline", StringType(), True),
        StructField("brief_description", StringType(), True),
        StructField("long_description", StringType(), True),
        StructField("price_cents", LongType(), True),
        StructField("currency", StringType(), True),
        StructField("inventory_count", LongType(), True),
        StructField("primary_category", StringType(), True),
        StructField("main_image_path", StringType(), True),
        StructField("secondary_image_paths", StringType(), True),
        StructField("seo_slug", StringType(), True),
        StructField("weight_grams", LongType(), True),
        StructField("tags", StringType(), True),
        StructField("ds", StringType(), True),
    ]
)

USER_ACTIVITIES_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("event_time", TimestampType(), True),
        StructField("ingested_time", TimestampType(), True),
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("country_code", StringType(), True),
        StructField("listing_id", LongType(), True),
        StructField("event_type", StringType(), True),
        StructField("ds", StringType(), True),
    ]
)


def _ts(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def create_warehouse(spark: SparkSession) -> None:
    """Create namespaces and seed tables. Safe to call on an already-populated warehouse."""
    spark.sql("CREATE NAMESPACE IF NOT EXISTS demo")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS data")
    _create_dim_listings(spark)
    _create_user_activities(spark)


def _create_dim_listings(spark: SparkSession) -> None:
    rows = [
        (
            1,
            1,
            1735603260000,
            1735603260000,
            True,
            "listing",
            "zero listing",
            "mega listing",
            10,
            "USD",
            1,
            "bed",
            "someimage",
            "second_image",
            "slug",
            100,
            "tag",
            "2024-12-31",
        ),  # noqa: E241
        (
            1,
            1,
            1735603260000,
            1735693320000,
            True,
            "listing",
            "first listing",
            "mega listing",
            10,
            "USD",
            1,
            "bed",
            "someimage",
            "second_image",
            "slug",
            100,
            "tag",
            "2025-01-01",
        ),  # noqa: E241
        (
            1,
            1,
            1735603260000,
            1735779780000,
            True,
            "listing",
            "first listing update",
            "mega listing",
            100,
            "USD",
            1,
            "bed",
            "someimage",
            "second_image",
            "slug",
            100,
            "tag",
            "2025-01-02",
        ),  # noqa: E241
        (
            1,
            1,
            1735603260000,
            1735866300000,
            True,
            "listing",
            "second listing update",
            "mega listing",
            1000,
            "USD",
            1,
            "bed",
            "someimage",
            "second_image",
            "slug",
            100,
            "tag",
            "2025-01-03",
        ),  # noqa: E241
    ]
    df = spark.createDataFrame(rows, DIM_LISTINGS_SCHEMA)
    df.writeTo("demo.dim_listings").using("iceberg").partitionedBy("ds").createOrReplace()


def _create_user_activities(spark: SparkSession) -> None:
    rows = [
        (
            "e0",
            _ts(2025, 1, 1, 0, 1),
            _ts(2025, 1, 1, 0, 10),
            "user_1",
            "session_1",
            "mobile",
            "CL",
            1,
            "view",
            "2025-01-01",
        ),
        (
            "e1",
            _ts(2025, 1, 2, 0, 1),
            _ts(2025, 1, 2, 0, 10),
            "user_1",
            "session_2",
            "mobile",
            "CL",
            1,
            "view",
            "2025-01-02",
        ),
        (
            "e2",
            _ts(2025, 1, 2, 0, 2),
            _ts(2025, 1, 2, 0, 11),
            "user_1",
            "session_3",
            "mobile",
            "CL",
            1,
            "view",
            "2025-01-02",
        ),
        (
            "e3",
            _ts(2025, 1, 3, 0, 3),
            _ts(2025, 1, 3, 0, 9),
            "user_1",
            "session_4",
            "mobile",
            "CL",
            1,
            "view",
            "2025-01-03",
        ),
        (
            "e4",
            _ts(2025, 1, 3, 0, 3, 10),
            _ts(2025, 1, 3, 0, 8, 10),
            "user_1",
            "session_5",
            "mobile",
            "CL",
            1,
            "view",
            "2025-01-03",
        ),
    ]
    df = spark.createDataFrame(rows, USER_ACTIVITIES_SCHEMA)
    df.writeTo("demo.user_activities").using("iceberg").partitionedBy("ds").createOrReplace()
