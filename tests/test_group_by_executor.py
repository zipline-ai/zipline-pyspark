from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.jupyter.group_by_executor import JupyterGroupBy


def _make_gb(name="test_gb", namespace="test_ns"):
    gb = MagicMock()
    gb.metaData.name = name
    gb.metaData.outputNamespace = namespace
    return gb


class TestOutputTable:
    def test_simple(self):
        jgb = JupyterGroupBy(_make_gb(), MagicMock())
        assert jgb.output_table == "test_ns.test_gb"

    def test_name_sanitized(self):
        jgb = JupyterGroupBy(_make_gb(name="my.group-by", namespace="ns"), MagicMock())
        assert jgb.output_table == "ns.my_group_by"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.side_effect = lambda t, n: [None] * n
        return spark

    def _get_cli_args(self, spark):
        return list(spark._jvm.ai.chronon.spark.Driver.main.call_args[0][0])

    def test_subcommand_is_group_by_backfill(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")

        args = self._get_cli_args(spark)
        assert "group-by-backfill" in args

    def test_no_exit_included(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")

        args = self._get_cli_args(spark)
        assert "--no-exit" in args

    def test_conf_path_and_end_date_included(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")

        args = self._get_cli_args(spark)
        assert "--conf-path" in args
        assert "/tmp/conf.json" in args
        assert "--end-date" in args
        assert "2026-04-07" in args

    def test_start_partition_always_included(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")

        args = self._get_cli_args(spark)
        assert "--start-partition" in args
        assert "2026-04-01" in args

    def test_step_days_included_when_set(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01", step_days=3)

        args = self._get_cli_args(spark)
        assert "--step-days" in args
        assert "3" in args

    def test_step_days_omitted_when_none(self):
        spark = self._make_spark()
        jgb = JupyterGroupBy(_make_gb(), spark)
        jgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")

        args = self._get_cli_args(spark)
        assert "--step-days" not in args

    def test_raises_runtime_error_when_jar_missing(self, spark, tmp_path):
        gb = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.gb__0", outputNamespace="data")
        )
        (tmp_path / "group_by.json").write_text("{}")
        jgb = JupyterGroupBy(gb, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            jgb._invoke_driver(str(tmp_path / "group_by.json"), "2026-01-01", "2025-12-01")


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        step_days=None,
        run_first_hole=True,
    )

    def _run(self, gb, spark, tmp_path, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-07")
        start_date = run_kwargs.pop("start_date", "2026-04-01")
        jgb = JupyterGroupBy(gb, spark, tmp_dir=str(tmp_path))
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "group_by.json")

        with (
            patch(
                "ai.chronon.pyspark.jupyter.session.ChrononSession.compile_group_by_to_file"
            ) as mock_compile,
            patch.object(jgb, "_invoke_driver") as mock_invoke,
        ):
            result = jgb.run(end_date, start_date, **run_kwargs)

        return result, expected_df, mock_invoke, mock_compile, conf_path

    def test_compile_called_with_correct_path(self, tmp_path):
        spark = MagicMock()
        _, _, _, mock_compile, conf_path = self._run(_make_gb(), spark, tmp_path)
        assert mock_compile.call_args[0][2] == conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_gb(), spark, tmp_path)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", "2026-04-01", **self._DEFAULT_INVOKE_KWARGS
        )

    def test_step_days_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_gb(), spark, tmp_path, step_days=3)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "step_days": 3}
        )

    def test_run_first_hole_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_gb(), spark, tmp_path, run_first_hole=False
        )
        mock_invoke.assert_called_once_with(
            conf_path,
            "2026-04-07",
            "2026-04-01",
            **{**self._DEFAULT_INVOKE_KWARGS, "run_first_hole": False},
        )

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _, _ = self._run(_make_gb(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_gb")
        assert result is expected_df

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        jgb = JupyterGroupBy(_make_gb(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_group_by_to_file"),
            patch.object(jgb, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            jgb.run("2026-04-07", "2026-04-01")

        mock_mkdtemp.assert_called_once()
