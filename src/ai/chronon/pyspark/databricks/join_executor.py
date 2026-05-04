import logging
import os
import tempfile
from typing import Any, Callable, Optional

from pyspark.sql import DataFrame, SparkSession

from ai.chronon.pyspark.jupyter.session import ChrononSession
from ai.chronon.pyspark.jupyter.staging_query_executor import _sanitize

logger = logging.getLogger(__name__)


class DatabricksJoin:
    """Executes a Chronon Join backfill on Databricks by delegating to the Scala batch driver.

    Mirrors JupyterJoin with additional Databricks notebook integrations:
    an optional display function for rich DataFrame rendering and dbutils-backed
    widgets for interactive date inputs.

    Args:
        join: Thrift Join object. ``metaData.name`` and ``metaData.team``
            must be set before calling ``run()``.
        spark: Active SparkSession (the ``spark`` global in Databricks notebooks).
        chronon_root: Path to the chronon config repo root (where ``teams.py`` lives).
            Defaults to the ``CHRONON_ROOT`` env var or the current working directory.
        tmp_dir: Directory for the compiled config file. A fresh ``tempfile.mkdtemp``
            is used when omitted.
        dbutils: Databricks ``dbutils`` object, used by ``setup_widgets()`` and
            ``get_widget_dates()``.
        display_fn: Callable invoked with the output DataFrame at the end of ``run()``.
            Pass the notebook-scoped ``display`` function to render results inline.
    """

    def __init__(
        self,
        join,
        spark: SparkSession,
        chronon_root: Optional[str] = None,
        tmp_dir: Optional[str] = None,
        dbutils: Optional[Any] = None,
        display_fn: Optional[Callable[[DataFrame], None]] = None,
    ):
        self.join = join
        self.spark = spark
        self._chronon_root = chronon_root or os.getenv("CHRONON_ROOT", os.getcwd())
        self._tmp_dir = tmp_dir
        self._dbutils = dbutils
        self._display_fn = display_fn

    @property
    def output_table(self) -> str:
        """Fully-qualified output table name derived from the config metadata."""
        meta = self.join.metaData
        return f"{meta.outputNamespace}.{_sanitize(meta.name)}"

    def setup_widgets(self, default_end_date: str = "", default_start_date: str = "") -> None:
        """Create Databricks text widgets for ``end_date`` and ``start_date`` inputs."""
        if self._dbutils is None:
            raise RuntimeError("dbutils must be provided to use setup_widgets()")
        self._dbutils.widgets.text("end_date", default_end_date, "End Date (YYYY-MM-DD)")
        self._dbutils.widgets.text(
            "start_date", default_start_date, "Start Date (YYYY-MM-DD, optional)"
        )

    def get_widget_dates(self):
        """Return ``(end_date, start_date)`` from Databricks widgets.

        ``start_date`` is ``None`` when the widget is left blank.
        """
        if self._dbutils is None:
            raise RuntimeError("dbutils must be provided to use get_widget_dates()")
        end_date = self._dbutils.widgets.get("end_date")
        raw_start = self._dbutils.widgets.get("start_date")
        return end_date, raw_start or None

    def _invoke_driver(
        self,
        conf_path: str,
        end_date: str,
        start_date: Optional[str] = None,
        run_first_hole: bool = True,
    ) -> None:
        """Call ai.chronon.spark.Driver.main() with the join subcommand."""
        gateway = self.spark.sparkContext._gateway
        jvm = self.spark._jvm

        cli_args = [
            "--no-exit",
            "join",
            "--conf-path",
            conf_path,
            "--end-date",
            end_date,
        ]
        if run_first_hole:
            cli_args.append("--run-first-hole")
        if start_date is not None:
            cli_args += ["--start-partition", start_date]

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
        run_first_hole: bool = True,
    ) -> DataFrame:
        """Compile the config and run the Scala join driver.

        If a ``display_fn`` was supplied at construction time it is called with the
        output DataFrame before returning, enabling inline notebook rendering.

        Args:
            end_date: Inclusive end partition (YYYY-MM-DD or YYYYMMDD).
            start_date: Optional inclusive start partition. When omitted the driver
                determines the start from the join's left source startPartition.
            run_first_hole: Fill the first unfilled partition range even if later
                partitions already exist.
        """
        tmp_dir = self._tmp_dir or tempfile.mkdtemp(prefix="chronon_join_")
        conf_path = os.path.join(tmp_dir, "join.json")
        ChrononSession.compile_join_to_file(self.join, self._chronon_root, conf_path)
        ChrononSession.apply_execution_conf(self.spark, self.join)

        self._invoke_driver(
            conf_path,
            end_date,
            start_date=start_date,
            run_first_hole=run_first_hole,
        )

        df = self.spark.table(self.output_table)
        if self._display_fn is not None:
            self._display_fn(df)
        return df
