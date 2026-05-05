from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.databricks.group_by_executor import DatabricksGroupBy


def _make_gb(name="test_gb", namespace="test_ns"):
    gb = MagicMock()
    gb.metaData.name = name
    gb.metaData.outputNamespace = namespace
    return gb


class TestOutputTable:
    def test_simple(self):
        dgb = DatabricksGroupBy(_make_gb(), MagicMock())
        assert dgb.output_table == "test_ns.test_gb"

    def test_name_sanitized(self):
        dgb = DatabricksGroupBy(_make_gb(name="my.group-by", namespace="ns"), MagicMock())
        assert dgb.output_table == "ns.my_group_by"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.side_effect = lambda t, n: [None] * n
        return spark

    def _get_cli_args(self, spark):
        return list(spark._jvm.ai.chronon.spark.Driver.main.call_args[0][0])

    def test_subcommand_is_group_by_backfill(self):
        spark = self._make_spark()
        dgb = DatabricksGroupBy(_make_gb(), spark)
        dgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")
        args = self._get_cli_args(spark)
        assert "group-by-backfill" in args

    def test_start_partition_always_included(self):
        spark = self._make_spark()
        dgb = DatabricksGroupBy(_make_gb(), spark)
        dgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")
        args = self._get_cli_args(spark)
        assert "--start-partition" in args
        assert "2026-04-01" in args

    def test_step_days_included_when_set(self):
        spark = self._make_spark()
        dgb = DatabricksGroupBy(_make_gb(), spark)
        dgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01", step_days=3)
        args = self._get_cli_args(spark)
        assert "--step-days" in args
        assert "3" in args

    def test_step_days_omitted_when_none(self):
        spark = self._make_spark()
        dgb = DatabricksGroupBy(_make_gb(), spark)
        dgb._invoke_driver("/tmp/conf.json", "2026-04-07", "2026-04-01")
        args = self._get_cli_args(spark)
        assert "--step-days" not in args

    def test_raises_when_jar_not_on_classpath(self, spark, tmp_path):
        gb = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.gb__0", outputNamespace="data")
        )
        (tmp_path / "group_by.json").write_text("{}")
        dgb = DatabricksGroupBy(gb, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            dgb._invoke_driver(str(tmp_path / "group_by.json"), "2026-01-01", "2025-12-01")


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        step_days=None,
        run_first_hole=True,
    )

    def _run(self, gb, spark, tmp_path, display_fn=None, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-07")
        start_date = run_kwargs.pop("start_date", "2026-04-01")
        dgb = DatabricksGroupBy(gb, spark, tmp_dir=str(tmp_path), display_fn=display_fn)
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "group_by.json")

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_group_by_to_file"),
            patch.object(dgb, "_invoke_driver") as mock_invoke,
        ):
            result = dgb.run(end_date, start_date, **run_kwargs)

        return result, expected_df, mock_invoke, conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(_make_gb(), spark, tmp_path)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", "2026-04-01", **self._DEFAULT_INVOKE_KWARGS
        )

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _ = self._run(_make_gb(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_gb")
        assert result is expected_df

    def test_display_fn_called_with_dataframe(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        result, expected_df, _, _ = self._run(_make_gb(), spark, tmp_path, display_fn=display_fn)
        display_fn.assert_called_once_with(expected_df)
        assert result is expected_df

    def test_display_fn_not_called_when_none(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        self._run(_make_gb(), spark, tmp_path, display_fn=None)
        display_fn.assert_not_called()

    def test_step_days_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(_make_gb(), spark, tmp_path, step_days=3)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "step_days": 3}
        )

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        dgb = DatabricksGroupBy(_make_gb(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_group_by_to_file"),
            patch.object(dgb, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            dgb.run("2026-04-07", "2026-04-01")

        mock_mkdtemp.assert_called_once()


class TestWidgets:
    def test_setup_widgets_calls_dbutils(self):
        dbutils = MagicMock()
        dgb = DatabricksGroupBy(_make_gb(), MagicMock(), dbutils=dbutils)
        dgb.setup_widgets(default_end_date="2026-04-07", default_start_date="2026-04-01")
        assert dbutils.widgets.text.call_count == 2

    def test_setup_widgets_raises_without_dbutils(self):
        dgb = DatabricksGroupBy(_make_gb(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dgb.setup_widgets()

    def test_get_widget_dates_returns_values(self):
        dbutils = MagicMock()
        dbutils.widgets.get.side_effect = lambda key: (
            "2026-04-07" if key == "end_date" else "2026-04-01"
        )
        dgb = DatabricksGroupBy(_make_gb(), MagicMock(), dbutils=dbutils)
        end_date, start_date = dgb.get_widget_dates()
        assert end_date == "2026-04-07"
        assert start_date == "2026-04-01"

    def test_get_widget_dates_raises_without_dbutils(self):
        dgb = DatabricksGroupBy(_make_gb(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dgb.get_widget_dates()
