"""Session-scoped Spark fixture for integration tests.

Each integration test run uses a single SparkSession with one cloud JAR loaded.
PySpark's SparkContext is a JVM singleton — multiple live sessions in the same
process are not supported — so all tests in this directory share one session.

Which JAR to load
-----------------
Pass ``--cloud {aws,gcp,azure}`` to pytest, or let the fixture auto-select
the first JAR it finds in tests/resources/jars/.  Tests are skipped (not
failed) when the requested JAR is absent.

Running integration tests in isolation
---------------------------------------
Always run integration tests in their own pytest invocation so the unit-test
session's plain SparkSession (no JAR, no Iceberg) does not collide::

    pytest tests/integration/ --cloud aws
    pytest tests/integration/           # auto-selects first available cloud

Do NOT mix with the top-level ``pytest tests/`` invocation unless you are
sure the integration session starts first.
"""

import logging
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from tests.integration.warehouse import create_warehouse

JARS_DIR = Path(__file__).parent.parent / "resources" / "jars"
_CLOUDS = ["aws", "gcp", "azure"]


def pytest_addoption(parser):
    parser.addoption(
        "--cloud",
        default=None,
        choices=_CLOUDS,
        help="Cloud provider JAR to load for integration tests (default: first available)",
    )


@pytest.fixture(scope="session")
def cloud(request):
    explicit = request.config.getoption("--cloud")
    if explicit:
        return explicit
    for candidate in _CLOUDS:
        if (JARS_DIR / f"cloud_{candidate}.jar").exists():
            return candidate
    pytest.skip("No cloud JAR found in tests/resources/jars/ — see .gitkeep for instructions")


@pytest.fixture(scope="session")
def warehouse_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("iceberg_warehouse")


@pytest.fixture(scope="session")
def spark(cloud, warehouse_dir):
    jar_path = JARS_DIR / f"cloud_{cloud}.jar"
    if not jar_path.exists():
        pytest.skip(f"JAR not found: {jar_path}")

    session = (
        SparkSession.builder.master("local[2]")
        .appName(f"zipline-integration-{cloud}")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.jars", str(jar_path))
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
        .config("spark.sql.catalog.spark_catalog.type", "hadoop")
        .config("spark.sql.catalog.spark_catalog.warehouse", str(warehouse_dir))
        .getOrCreate()
    )

    logging.getLogger("py4j").setLevel(logging.WARNING)
    create_warehouse(session)
    yield session
    session.stop()
