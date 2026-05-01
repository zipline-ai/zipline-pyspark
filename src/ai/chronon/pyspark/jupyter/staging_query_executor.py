import os
import re
import tempfile
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from ai.chronon.pyspark.jupyter.session import ChrononSession


def _parse_date(ds: str) -> datetime:
    """Accept YYYYMMDD or YYYY-MM-DD and return a datetime."""
    ds = ds.strip()
    if re.fullmatch(r"\d{8}", ds):
        return datetime.strptime(ds, "%Y%m%d")
    return datetime.strptime(ds, "%Y-%m-%d")


def _render_query(query: str, start_date: str, end_date: str) -> str:
    """Substitute {{ start_date }} and {{ end_date }} placeholders in the query."""
    result = re.sub(r"\{\{\s*start_date(?:\([^)]*\))?\s*\}\}", start_date, query)
    result = re.sub(r"\{\{\s*end_date(?:\([^)]*\))?\s*\}\}", end_date, result)
    return result


def _sanitize(name: Optional[str]) -> Optional[str]:
    """Mirror of Scala's MetaData.cleanName — replaces non-alphanumeric chars with _."""
    if name is None:
        return None
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class JupyterStagingQuery:
    """Executes a Chronon StagingQuery directly against a PySpark session.

    Intended for interactive use in Jupyter notebooks where running a full
    spark-submit job is impractical.  Handles date-range chunking and
    query template rendering; the full upstream auto-expand logic from
    the Chronon Spark Driver is not replicated here.
    """

    def __init__(self, staging_query, spark: SparkSession):
        self.staging_query = staging_query
        self.spark = spark

    def run(
        self,
        end_date: str,
        step_days: int = 1,
        enable_auto_expand: bool = False,
        start_date: Optional[str] = None,
    ) -> DataFrame:
        """Execute the staging query and return the result as a single DataFrame.

        Dates can be supplied in YYYYMMDD or YYYY-MM-DD format.

        When step_days > 1 each chunk is executed separately and the results
        are union-ed together.  When enable_auto_expand is True the start date
        is extended back by one step from the nominal start so that the first
        step picks up any newly available upstream partitions.

        Args:
            end_date: Inclusive end date.
            step_days: Days per execution step.
            enable_auto_expand: Extend the date range one extra step at the
                start to incorporate newly available upstream data.
            start_date: Explicit start date.  When omitted the query is run
                as a single step covering end_date only.
        """
        end_dt = _parse_date(end_date)

        if start_date is not None:
            start_dt = _parse_date(start_date)
        else:
            start_dt = end_dt

        if enable_auto_expand:
            start_dt = start_dt - timedelta(days=step_days)

        for stmt in self.staging_query.setups or []:
            self.spark.sql(stmt)

        windows = []
        current = start_dt
        while current <= end_dt:
            step_end = min(current + timedelta(days=step_days - 1), end_dt)
            windows.append((current, step_end))
            current = step_end + timedelta(days=1)

        dfs = []
        for step_start, step_end in windows:
            query = _render_query(
                self.staging_query.query,
                step_start.strftime("%Y-%m-%d"),
                step_end.strftime("%Y-%m-%d"),
            )
            dfs.append(self.spark.sql(query))

        if not dfs:
            return self.spark.createDataFrame([], StructType([]))

        return reduce(DataFrame.union, dfs)


class BatchStagingQuery:
    """Executes a Chronon StagingQuery by delegating to the Scala batch driver.

    Compiles the staging-query config into ``tmp_dir`` (running the same namespace
    propagation and team-conf merging as the CLI compile step), then invokes
    ``ai.chronon.spark.batch.StagingQuery.main()`` via the Py4J gateway and reads the
    output table back as a DataFrame.

    Requires the Chronon batch JAR to be on the SparkSession classpath.

    Args:
        staging_query: Thrift StagingQuery object. ``metaData.name`` and
            ``metaData.team`` must be set before calling ``run()``.
        spark: Active SparkSession.
        chronon_root: Path to the chronon config repo root (where ``teams.py`` lives).
            Defaults to the ``CHRONON_ROOT`` env var or the current working directory.
        tmp_dir: Directory for the compiled config file.  A fresh ``tempfile.mkdtemp``
            is used when omitted.
    """

    def __init__(
        self,
        staging_query,
        spark: SparkSession,
        chronon_root: Optional[str] = None,
        tmp_dir: Optional[str] = None,
    ):
        self.staging_query = staging_query
        self.spark = spark
        self._chronon_root = chronon_root or os.getenv("CHRONON_ROOT", os.getcwd())
        self._tmp_dir = tmp_dir

    @property
    def output_table(self) -> str:
        """Fully-qualified output table name derived from the config metadata."""
        meta = self.staging_query.metaData
        return f"{meta.outputNamespace}.{_sanitize(meta.name)}"

    def _invoke_driver(self, conf_path: str, end_date: str, step_days: Optional[int]) -> None:
        """Call ai.chronon.spark.batch.StagingQuery.main() via the Py4J gateway."""
        gateway = self.spark.sparkContext._gateway
        jvm = self.spark._jvm

        cli_args = ["--conf-path", conf_path, "--end-date", end_date]
        if step_days is not None:
            cli_args += ["--step-days", str(step_days)]

        java_args = gateway.new_array(jvm.String, len(cli_args))
        for i, arg in enumerate(cli_args):
            java_args[i] = arg

        jvm.ai.chronon.spark.batch.StagingQuery.main(java_args)

    def run(
        self,
        end_date: str,
        step_days: Optional[int] = None,
    ) -> DataFrame:
        """Compile the config, run the Scala driver, and return the output table.

        Args:
            end_date: Inclusive end partition (YYYY-MM-DD or YYYYMMDD).
            step_days: Optional maximum step size in days passed to the driver.
        """
        end_date = _parse_date(end_date).strftime("%Y-%m-%d")

        tmp_dir = self._tmp_dir or tempfile.mkdtemp(prefix="chronon_staging_query_")
        conf_path = os.path.join(tmp_dir, "staging_query.json")
        ChrononSession.compile_to_file(self.staging_query, self._chronon_root, conf_path)

        self._invoke_driver(conf_path, end_date, step_days)

        return self.spark.table(self.output_table)
