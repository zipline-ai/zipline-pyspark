"""Integration tests for JupyterStagingQuery against a local Iceberg warehouse.

These tests load the cloud-specific Chronon JAR and execute real Spark SQL through the
Scala batch driver.  They require the warehouse tables created by conftest.py and skip
automatically when the JAR is absent.

``compile_to_file`` is patched out because the compile step (which needs zipline-ai's
CLI machinery) is covered by unit tests.  What we test here is the full
JAR invocation + Iceberg read/write path.

Each test writes to a unique output table (derived from pytest's tmp_path) so that
tests are isolated even within a shared session-scoped SparkSession.
"""

import json
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ai.chronon.pyspark.jupyter.staging_query_executor import JupyterStagingQuery

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

_DIM_LISTINGS_QUERY = """
SELECT *
FROM demo.dim_listings
WHERE ds BETWEEN {{ start_date }} AND {{ end_date }}
"""

# Explicit column list avoids ambiguity: demo.user_activities already has a
# ds STRING column; we recompute ds from event_time and drop the source ds.
_USER_ACTIVITIES_QUERY = """
SELECT
    event_id, user_id, session_id, device_type, country_code,
    listing_id, event_type, event_time,
    DATE_FORMAT(event_time, 'yyyy-MM-dd') as ds
FROM demo.user_activities
WHERE event_time BETWEEN {{ start_date }} AND {{ end_date }}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_suffix(tmp_path) -> str:
    """Short, filesystem-safe suffix unique to this test's tmp_path."""
    raw = re.sub(r"[^a-z0-9]", "_", tmp_path.name.lower())
    return raw[:24]


def _conf_json(cloud: str, name: str, query: str, partition_col: str = "ds") -> str:
    """Minimal compiled-config JSON that the Scala StagingQuery driver can parse."""
    return json.dumps(
        {
            "engineType": 0,  # SPARK
            "metaData": {
                "name": f"{cloud}.integration_test.{name}__0",
                "outputNamespace": "data",
                "team": cloud,
                "executionInfo": {
                    "stepDays": 10,
                    "conf": {
                        "common": {
                            "spark.chronon.partition.column": "ds",
                            "spark.chronon.partition.format": "yyyy-MM-dd",
                            "spark.chronon.table_write.format": "iceberg",
                            "spark.sql.shuffle.partitions": "2",
                        }
                    },
                },
            },
            "query": query,
            "tableDependencies": [
                {
                    "endOffset": {"length": 0, "timeUnit": 1},
                    "startOffset": {"length": 0, "timeUnit": 1},
                    "tableInfo": {
                        "partitionColumn": partition_col,
                        "partitionInterval": {"length": 1, "timeUnit": 1},
                        "table": f"demo.{name}",
                    },
                }
            ],
        }
    )


def _make_sq(cloud: str, name: str, extra_conf: dict = None) -> object:
    """Minimal staging-query stub: only the fields JupyterStagingQuery reads directly."""
    conf = SimpleNamespace(common=extra_conf or {}, modeConfigs=None)
    execution_info = SimpleNamespace(conf=conf)
    return SimpleNamespace(
        metaData=SimpleNamespace(
            name=f"{cloud}.integration_test.{name}__0",
            outputNamespace="data",
            executionInfo=execution_info,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExecutionConfApplied:
    def test_conf_from_staging_query_set_on_session(self, spark, cloud, tmp_path):
        suffix = _table_suffix(tmp_path)
        name = f"dim_listings_{suffix}"
        conf_path = tmp_path / "staging_query.json"
        conf_path.write_text(_conf_json(cloud, name, _DIM_LISTINGS_QUERY))

        sq = _make_sq(cloud, name, extra_conf={"spark.chronon.integration.test": "applied"})
        jsq = JupyterStagingQuery(sq, spark, tmp_dir=str(tmp_path))

        with patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"):
            jsq.run("2025-01-03", start_date="2025-01-01", step_days=10)

        assert spark.conf.get("spark.chronon.integration.test") == "applied"


@pytest.mark.integration
class TestDimListingsExport:
    def _run(
        self, spark, cloud, tmp_path, end_date="2025-01-03", start_date="2025-01-01", step_days=10
    ):
        suffix = _table_suffix(tmp_path)
        name = f"dim_listings_{suffix}"
        conf_path = tmp_path / "staging_query.json"
        conf_path.write_text(_conf_json(cloud, name, _DIM_LISTINGS_QUERY))

        jsq = JupyterStagingQuery(_make_sq(cloud, name), spark, tmp_dir=str(tmp_path))

        with patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"):
            return jsq.run(end_date, start_date=start_date, step_days=step_days)

    def test_output_table_has_rows(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path)
        assert result.count() > 0

    def test_output_schema_preserves_source_columns(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path)
        assert "listing_id" in result.columns
        assert "ds" in result.columns
        assert "price_cents" in result.columns

    def test_output_respects_end_date(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path, end_date="2025-01-02")
        dates = {row["ds"] for row in result.select("ds").collect()}
        assert all(d <= "2025-01-02" for d in dates), f"Unexpected dates after end_date: {dates}"

    def test_output_respects_start_date(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path, start_date="2025-01-02", end_date="2025-01-03")
        dates = {row["ds"] for row in result.select("ds").collect()}
        assert all(d >= "2025-01-02" for d in dates), f"Unexpected dates before start_date: {dates}"


@pytest.mark.integration
class TestUserActivitiesExport:
    def _run(
        self, spark, cloud, tmp_path, end_date="2025-01-03", start_date="2025-01-01", step_days=10
    ):
        suffix = _table_suffix(tmp_path)
        name = f"user_activities_{suffix}"
        conf_path = tmp_path / "staging_query.json"
        conf_path.write_text(_conf_json(cloud, name, _USER_ACTIVITIES_QUERY, "event_time"))

        jsq = JupyterStagingQuery(_make_sq(cloud, name), spark, tmp_dir=str(tmp_path))

        with patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"):
            return jsq.run(end_date, start_date=start_date, step_days=step_days)

    def test_output_table_has_rows(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path)
        assert result.count() > 0

    def test_ds_column_derived_from_event_time(self, spark, cloud, tmp_path):
        result = self._run(spark, cloud, tmp_path)
        assert "ds" in result.columns
        dates = {row["ds"] for row in result.select("ds").collect()}
        assert dates <= {"2025-01-01", "2025-01-02", "2025-01-03"}
