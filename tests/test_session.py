import sys
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
