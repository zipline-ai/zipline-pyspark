import os
import re
import tempfile
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from ai.chronon.pyspark.jupyter.session import ChrononSession


def _sanitize(name: Optional[str]) -> Optional[str]:
    """Mirror of Scala's MetaData.cleanName — replaces non-alphanumeric chars with _."""
    if name is None:
        return None
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class JupyterStagingQuery:
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
        tmp_dir = self._tmp_dir or tempfile.mkdtemp(prefix="chronon_staging_query_")
        conf_path = os.path.join(tmp_dir, "staging_query.json")
        ChrononSession.compile_to_file(self.staging_query, self._chronon_root, conf_path)

        self._invoke_driver(conf_path, end_date, step_days)

        return self.spark.table(self.output_table)
