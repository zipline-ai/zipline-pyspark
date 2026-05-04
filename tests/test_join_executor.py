from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.jupyter.join_executor import JupyterJoin


def _make_join(name="test_join", namespace="test_ns"):
    j = MagicMock()
    j.metaData.name = name
    j.metaData.outputNamespace = namespace
    return j


class TestOutputTable:
    def test_simple(self):
        jj = JupyterJoin(_make_join(), MagicMock())
        assert jj.output_table == "test_ns.test_join"

    def test_name_sanitized(self):
        jj = JupyterJoin(_make_join(name="my.join-v1", namespace="ns"), MagicMock())
        assert jj.output_table == "ns.my_join_v1"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.side_effect = lambda t, n: [None] * n
        return spark

    def _get_cli_args(self, spark):
        return list(spark._jvm.ai.chronon.spark.Driver.main.call_args[0][0])

    def test_subcommand_is_join(self):
        spark = self._make_spark()
        jj = JupyterJoin(_make_join(), spark)
        jj._invoke_driver("/tmp/conf.json", "2026-04-07")

        args = self._get_cli_args(spark)
        assert "join" in args

    def test_no_exit_included(self):
        spark = self._make_spark()
        jj = JupyterJoin(_make_join(), spark)
        jj._invoke_driver("/tmp/conf.json", "2026-04-07")

        args = self._get_cli_args(spark)
        assert "--no-exit" in args

    def test_conf_path_and_end_date_included(self):
        spark = self._make_spark()
        jj = JupyterJoin(_make_join(), spark)
        jj._invoke_driver("/tmp/conf.json", "2026-04-07")

        args = self._get_cli_args(spark)
        assert "--conf-path" in args
        assert "/tmp/conf.json" in args
        assert "--end-date" in args
        assert "2026-04-07" in args

    def test_start_partition_included_when_set(self):
        spark = self._make_spark()
        jj = JupyterJoin(_make_join(), spark)
        jj._invoke_driver("/tmp/conf.json", "2026-04-07", start_date="2026-04-01")

        args = self._get_cli_args(spark)
        assert "--start-partition" in args
        assert "2026-04-01" in args

    def test_start_partition_omitted_when_none(self):
        spark = self._make_spark()
        jj = JupyterJoin(_make_join(), spark)
        jj._invoke_driver("/tmp/conf.json", "2026-04-07")

        args = self._get_cli_args(spark)
        assert "--start-partition" not in args

    def test_raises_runtime_error_when_jar_missing(self, spark, tmp_path):
        join = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.join__0", outputNamespace="data")
        )
        (tmp_path / "join.json").write_text("{}")
        jj = JupyterJoin(join, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            jj._invoke_driver(str(tmp_path / "join.json"), "2026-01-01")


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        start_date=None,
        run_first_hole=True,
    )

    def _run(self, join_obj, spark, tmp_path, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-07")
        jj = JupyterJoin(join_obj, spark, tmp_dir=str(tmp_path))
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "join.json")

        with (
            patch(
                "ai.chronon.pyspark.jupyter.session.ChrononSession.compile_join_to_file"
            ) as mock_compile,
            patch.object(jj, "_invoke_driver") as mock_invoke,
        ):
            result = jj.run(end_date, **run_kwargs)

        return result, expected_df, mock_invoke, mock_compile, conf_path

    def test_compile_called_with_correct_path(self, tmp_path):
        spark = MagicMock()
        _, _, _, mock_compile, conf_path = self._run(_make_join(), spark, tmp_path)
        assert mock_compile.call_args[0][2] == conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_join(), spark, tmp_path)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-07", **self._DEFAULT_INVOKE_KWARGS)

    def test_start_date_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_join(), spark, tmp_path, start_date="2026-04-01"
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", **{**self._DEFAULT_INVOKE_KWARGS, "start_date": "2026-04-01"}
        )

    def test_run_first_hole_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_join(), spark, tmp_path, run_first_hole=False
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", **{**self._DEFAULT_INVOKE_KWARGS, "run_first_hole": False}
        )

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _, _ = self._run(_make_join(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_join")
        assert result is expected_df

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        jj = JupyterJoin(_make_join(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_join_to_file"),
            patch.object(jj, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            jj.run("2026-04-07")

        mock_mkdtemp.assert_called_once()
