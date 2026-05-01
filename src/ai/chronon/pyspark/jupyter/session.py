from typing import Dict, List, Optional

from pyspark.sql import SparkSession


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
    def compile_to_file(staging_query, chronon_root: str, conf_path: str) -> None:
        """Compile a staging query via the Chronon compile infrastructure and write to conf_path.

        Applies load_teams + update_metadata (namespace propagation, team conf/env merging)
        then serializes with thrift_simple_json — the same path as the CLI compile step but
        operating on an in-memory object.

        Requires staging_query.metaData.name and staging_query.metaData.team to be set.
        """
        from ai.chronon.cli.compile.parse_teams import load_teams, update_metadata
        from ai.chronon.cli.compile.serializer import thrift_simple_json

        teams_dict = load_teams(chronon_root, print=False)
        update_metadata(staging_query, teams_dict)
        with open(conf_path, "w") as f:
            f.write(thrift_simple_json(staging_query))
