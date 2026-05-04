import logging
from typing import Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

logger = logging.getLogger(__name__)


class ChrononSession:
    """Builds a SparkSession pre-loaded with Chronon JARs.

    Args:
        jars: Local paths or URIs of JARs to add to spark.jars.
        app_name: Spark application name.
        config: Additional Spark config key/value pairs.
    """

    def __init__(
        self,
        jars: List[str],
        app_name: str = "chronon",
        config: Optional[Dict[str, str]] = None,
    ):
        self.jars = jars
        self.app_name = app_name
        self.config = config or {}

    def build(self) -> SparkSession:
        """Return a SparkSession configured with the supplied JARs."""
        builder = SparkSession.builder.appName(self.app_name)
        if self.jars:
            builder = builder.config("spark.jars", ",".join(self.jars))
        for k, v in self.config.items():
            builder = builder.config(k, v)
        return builder.getOrCreate()

    @staticmethod
    def apply_execution_conf(spark: SparkSession, compiled_obj, mode: Optional[str] = None) -> None:
        """Apply Spark confs from a compiled Chronon object's executionInfo to *spark*.

        Reads ``metaData.executionInfo.conf.common`` and, when *mode* is given,
        ``conf.modeConfigs[mode]`` (mode-specific values win on conflicts).
        Settings that are immutable at runtime are logged as warnings and skipped.
        """
        try:
            conf = compiled_obj.metaData.executionInfo.conf
        except AttributeError:
            return
        if conf is None:
            return

        settings: Dict[str, str] = {}
        if conf.common:
            settings.update(conf.common)
        if mode and conf.modeConfigs:
            settings.update(conf.modeConfigs.get(mode, {}))

        for key, value in settings.items():
            try:
                spark.conf.set(key, value)
                logger.info("Applied Spark conf: %s = %s", key, value)
            except AnalysisException:
                logger.warning(
                    "Skipping immutable Spark conf %s (must be set at session creation)", key
                )

    @staticmethod
    def compile_to_file(staging_query, chronon_root: str, conf_path: str) -> None:
        """Compile a staging query via the Chronon compile infrastructure and write to conf_path.

        Applies load_teams + update_metadata (namespace propagation, team conf/env merging)
        then serializes with thrift_simple_json — the same path as the CLI compile step but
        operating on an in-memory object.

        Requires staging_query.metaData.name and staging_query.metaData.team to be set.
        """
        from ai.chronon.cli.compile.parse_teams import load_teams, update_metadata
        from ai.chronon.cli.compile.serializer import thrift_simple_json
        from ai.chronon.staging_query import _get_output_table_name

        teams_dict = load_teams(chronon_root, print=False)
        update_metadata(staging_query, teams_dict)
        if not staging_query.metaData.name:
            _get_output_table_name(staging_query)
        with open(conf_path, "w") as f:
            f.write(thrift_simple_json(staging_query))

    @staticmethod
    def compile_group_by_to_file(group_by, chronon_root: str, conf_path: str) -> None:
        """Compile a GroupBy config and write to conf_path.

        Applies load_teams + update_metadata then serializes with thrift_simple_json.
        Requires group_by.metaData.name and group_by.metaData.team to be set.
        """
        from ai.chronon.cli.compile.parse_teams import load_teams, update_metadata
        from ai.chronon.cli.compile.serializer import thrift_simple_json

        teams_dict = load_teams(chronon_root, print=False)
        update_metadata(group_by, teams_dict)
        with open(conf_path, "w") as f:
            f.write(thrift_simple_json(group_by))

    @staticmethod
    def compile_join_to_file(join, chronon_root: str, conf_path: str) -> None:
        """Compile a Join config and write to conf_path.

        Applies load_teams + update_metadata then serializes with thrift_simple_json.
        Requires join.metaData.name and join.metaData.team to be set.
        """
        from ai.chronon.cli.compile.parse_teams import load_teams, update_metadata
        from ai.chronon.cli.compile.serializer import thrift_simple_json

        teams_dict = load_teams(chronon_root, print=False)
        update_metadata(join, teams_dict)
        with open(conf_path, "w") as f:
            f.write(thrift_simple_json(join))
