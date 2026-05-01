import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai.chronon.pyspark.jupyter.session import ChrononSession


class TestChrononSessionBuild:
    def test_returns_spark_session(self, spark):
        session = ChrononSession(jars=[]).build()
        assert session is not None

    def test_jars_set_in_config(self, spark):
        jars = ["/path/to/a.jar", "/path/to/b.jar"]
        session = ChrononSession(jars=jars).build()
        assert session.conf.get("spark.jars") == ",".join(jars)

    def test_no_jars_skips_spark_jars_config(self):
        builder = MagicMock()
        builder.appName.return_value = builder
        builder.getOrCreate.return_value = MagicMock()
        with patch("ai.chronon.pyspark.jupyter.session.SparkSession") as mock_ss:
            mock_ss.builder = builder
            ChrononSession(jars=[]).build()
        builder.config.assert_not_called()

    def test_extra_config_applied(self, spark):
        session = ChrononSession(jars=[], config={"spark.sql.shuffle.partitions": "4"}).build()
        assert session.conf.get("spark.sql.shuffle.partitions") == "4"


class TestChrononSessionCompileToFile:
    def _mock_chronon_cli_modules(self, teams_dict, serialized):
        """Return a sys.modules patch dict that stubs out ai.chronon.cli compile modules."""
        mock_load_teams = MagicMock(return_value=teams_dict)
        mock_update_metadata = MagicMock()
        mock_thrift = MagicMock(return_value=serialized)

        mock_parse_teams = MagicMock()
        mock_parse_teams.load_teams = mock_load_teams
        mock_parse_teams.update_metadata = mock_update_metadata

        mock_serializer = MagicMock()
        mock_serializer.thrift_simple_json = mock_thrift

        return (
            {
                "ai.chronon.cli.compile.parse_teams": mock_parse_teams,
                "ai.chronon.cli.compile.serializer": mock_serializer,
            },
            mock_load_teams,
            mock_update_metadata,
            mock_thrift,
        )

    def test_writes_serialized_json(self, tmp_path):
        sq = MagicMock()
        conf_path = str(tmp_path / "out.json")
        modules, mock_load, mock_update, mock_thrift = self._mock_chronon_cli_modules(
            {}, '{"k":"v"}'
        )
        with patch.dict(sys.modules, modules):
            ChrononSession.compile_to_file(sq, "/chronon_root", conf_path)

        mock_load.assert_called_once_with("/chronon_root", print=False)
        mock_update.assert_called_once_with(sq, {})
        mock_thrift.assert_called_once_with(sq)
        with open(conf_path) as f:
            assert f.read() == '{"k":"v"}'

    def test_teams_dict_passed_to_update_metadata(self, tmp_path):
        sq = MagicMock()
        teams = {"team_a": MagicMock()}
        conf_path = str(tmp_path / "out.json")
        modules, _, mock_update, _ = self._mock_chronon_cli_modules(teams, "{}")
        with patch.dict(sys.modules, modules):
            ChrononSession.compile_to_file(sq, "/root", conf_path)
        mock_update.assert_called_once_with(sq, teams)


def _make_compiled_obj(common=None, mode_configs=None):
    """Build a minimal compiled-object stub with executionInfo.conf populated."""
    conf = SimpleNamespace(common=common, modeConfigs=mode_configs)
    execution_info = SimpleNamespace(conf=conf)
    meta = SimpleNamespace(executionInfo=execution_info)
    return SimpleNamespace(metaData=meta)


class TestChrononSessionApplyExecutionConf:
    def test_common_confs_applied_to_spark(self, spark):
        obj = _make_compiled_obj(common={"spark.sql.shuffle.partitions": "8"})
        ChrononSession.apply_execution_conf(spark, obj)
        assert spark.conf.get("spark.sql.shuffle.partitions") == "8"

    def test_multiple_common_confs_all_applied(self, spark):
        obj = _make_compiled_obj(
            common={"spark.sql.shuffle.partitions": "16", "spark.default.parallelism": "32"}
        )
        ChrononSession.apply_execution_conf(spark, obj)
        assert spark.conf.get("spark.sql.shuffle.partitions") == "16"
        assert spark.conf.get("spark.default.parallelism") == "32"

    def test_mode_conf_merged_and_overrides_common(self, spark):
        obj = _make_compiled_obj(
            common={"spark.sql.shuffle.partitions": "8", "spark.default.parallelism": "16"},
            mode_configs={"backfill": {"spark.sql.shuffle.partitions": "64"}},
        )
        ChrononSession.apply_execution_conf(spark, obj, mode="backfill")
        assert spark.conf.get("spark.sql.shuffle.partitions") == "64"
        assert spark.conf.get("spark.default.parallelism") == "16"

    def test_no_execution_info_is_noop(self, spark):
        obj = SimpleNamespace(metaData=SimpleNamespace())
        before = spark.conf.get("spark.sql.shuffle.partitions")
        ChrononSession.apply_execution_conf(spark, obj)
        assert spark.conf.get("spark.sql.shuffle.partitions") == before

    def test_none_conf_is_noop(self, spark):
        obj = _make_compiled_obj(common=None)
        before = spark.conf.get("spark.sql.shuffle.partitions")
        ChrononSession.apply_execution_conf(spark, obj)
        assert spark.conf.get("spark.sql.shuffle.partitions") == before

    def test_conf_object_none_is_noop(self, spark):
        execution_info = SimpleNamespace(conf=None)
        obj = SimpleNamespace(metaData=SimpleNamespace(executionInfo=execution_info))
        before = spark.conf.get("spark.sql.shuffle.partitions")
        ChrononSession.apply_execution_conf(spark, obj)
        assert spark.conf.get("spark.sql.shuffle.partitions") == before

    def test_immutable_conf_warns_and_continues(self, spark):
        from pyspark.sql.utils import AnalysisException

        mock_spark = MagicMock()
        mock_spark.conf.set.side_effect = [
            AnalysisException("immutable"),
            None,
        ]
        obj = _make_compiled_obj(
            common={"spark.master": "local", "spark.sql.shuffle.partitions": "8"}
        )
        ChrononSession.apply_execution_conf(mock_spark, obj)
        assert mock_spark.conf.set.call_count == 2
