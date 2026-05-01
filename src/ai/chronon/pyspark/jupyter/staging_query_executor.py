import logging
import os
import re
import tempfile
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from ai.chronon.pyspark.jupyter.session import ChrononSession

logger = logging.getLogger(__name__)


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

    def _invoke_driver(
        self,
        conf_path: str,
        end_date: str,
        start_date: Optional[str] = None,
        step_days: Optional[int] = None,
        enable_auto_expand: bool = True,
        force_overwrite: bool = False,
        run_first_hole: bool = True,
    ) -> None:
        """Call ai.chronon.spark.Driver.main() with the staging-query-backfill subcommand."""
        gateway = self.spark.sparkContext._gateway
        jvm = self.spark._jvm

        cli_args = [
            "staging-query-backfill",
            "--conf-path",
            conf_path,
            "--end-date",
            end_date,
        ]
        if enable_auto_expand:
            cli_args.append("--enable-auto-expand")
        if force_overwrite:
            cli_args.append("--force-overwrite")
        if run_first_hole:
            cli_args.append("--run-first-hole")
        if start_date is not None:
            cli_args += ["--start-partition", start_date]
        if step_days is not None:
            cli_args += ["--step-days", str(step_days)]

        logger.info("Invoking Scala driver: %s", " ".join(cli_args))
        java_args = gateway.new_array(jvm.String, len(cli_args))
        for i, arg in enumerate(cli_args):
            java_args[i] = arg

        try:
            jvm.ai.chronon.spark.Driver.main(java_args)
        except TypeError:
            conf = self.spark.sparkContext.getConf()
            jars = conf.get("spark.jars", "(not set)")
            extra_cp = conf.get("spark.driver.extraClassPath", "(not set)")
            raise RuntimeError(
                "Class ai.chronon.spark.Driver not found on the JVM classpath. "
                "Load the Chronon batch JAR via ChrononSession or spark.jars.\n"
                f"  spark.jars                    = {jars}\n"
                f"  spark.driver.extraClassPath   = {extra_cp}"
            ) from None

    def run(
        self,
        end_date: str,
        start_date: Optional[str] = None,
        step_days: Optional[int] = None,
        enable_auto_expand: bool = True,
        force_overwrite: bool = False,
        run_first_hole: bool = True,
    ) -> DataFrame:
        """
        Compile the config, run the Scala staging-query-backfill driver.

        Args:
            end_date: Inclusive end partition (YYYY-MM-DD or YYYYMMDD).
            start_date: Optional inclusive start partition. When omitted the driver
                determines the start from existing table state.
            step_days: Maximum days per step. Falls back to the value in executionInfo,
                then defaults to 30.
            enable_auto_expand: Auto-expand the output table when new columns appear.
            force_overwrite: Overwrite already-populated partitions.
            run_first_hole: When True (default), fill the first unfilled partition range
                even if later partitions already exist.
        """
        tmp_dir = self._tmp_dir or tempfile.mkdtemp(prefix="chronon_staging_query_")
        conf_path = os.path.join(tmp_dir, "staging_query.json")
        ChrononSession.compile_to_file(self.staging_query, self._chronon_root, conf_path)

        self._invoke_driver(
            conf_path,
            end_date,
            start_date=start_date,
            step_days=step_days,
            enable_auto_expand=enable_auto_expand,
            force_overwrite=force_overwrite,
            run_first_hole=run_first_hole,
        )

        return self.spark.table(self.output_table)
