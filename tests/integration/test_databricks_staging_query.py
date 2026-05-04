"""Integration tests for DatabricksStagingQuery against a local Iceberg warehouse.

The execution path (compile → Scala driver → Iceberg read/write) is identical to
JupyterStagingQuery and is already exercised by test_staging_query.py.  These tests
focus on the Databricks-specific layer: that display_fn receives the real result
DataFrame and that the executor behaves correctly with and without it.

No Databricks account or API token is required.  Tests run against the same local
JAR + Iceberg warehouse used by the rest of the integration suite.
"""

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.databricks.staging_query_executor import DatabricksStagingQuery

# ---------------------------------------------------------------------------
# Helpers (identical to test_staging_query.py)
# ---------------------------------------------------------------------------


def _table_suffix(tmp_path) -> str:
    raw = re.sub(r"[^a-z0-9]", "_", tmp_path.name.lower())
    return raw[:24]


def _conf_json(cloud: str, name: str, query: str) -> str:
    return json.dumps(
        {
            "engineType": 0,
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
                        "partitionColumn": "ds",
                        "partitionInterval": {"length": 1, "timeUnit": 1},
                        "table": f"demo.{name}",
                    },
                }
            ],
        }
    )


_DIM_LISTINGS_QUERY = """
SELECT *
FROM demo.dim_listings
WHERE ds BETWEEN {{ start_date }} AND {{ end_date }}
"""


def _make_sq(cloud: str, name: str) -> object:
    conf = SimpleNamespace(common={}, modeConfigs=None)
    return SimpleNamespace(
        metaData=SimpleNamespace(
            name=f"{cloud}.integration_test.{name}__0",
            outputNamespace="data",
            executionInfo=SimpleNamespace(conf=conf),
        )
    )


def _run(spark, cloud, tmp_path, display_fn=None, **run_kwargs):
    suffix = _table_suffix(tmp_path)
    name = f"db_dim_listings_{suffix}"
    (tmp_path / "staging_query.json").write_text(_conf_json(cloud, name, _DIM_LISTINGS_QUERY))

    dsq = DatabricksStagingQuery(
        _make_sq(cloud, name), spark, tmp_dir=str(tmp_path), display_fn=display_fn
    )
    with patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"):
        return dsq.run("2025-01-03", start_date="2025-01-01", **run_kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDisplayFn:
    def test_display_fn_called_with_result_dataframe(self, spark, cloud, tmp_path):
        display_fn = MagicMock()
        result = _run(spark, cloud, tmp_path, display_fn=display_fn)

        display_fn.assert_called_once()
        passed_df = display_fn.call_args[0][0]
        assert passed_df is result

    def test_display_fn_receives_non_empty_dataframe(self, spark, cloud, tmp_path):
        display_fn = MagicMock()
        _run(spark, cloud, tmp_path, display_fn=display_fn)

        passed_df = display_fn.call_args[0][0]
        assert passed_df.count() > 0

    def test_run_without_display_fn_still_returns_dataframe(self, spark, cloud, tmp_path):
        result = _run(spark, cloud, tmp_path, display_fn=None)
        assert result.count() > 0
