from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.databricks.staging_query_executor import DatabricksStagingQuery


def _make_sq(name="test_sq", namespace="test_ns"):
    sq = MagicMock()
    sq.metaData.name = name
    sq.metaData.outputNamespace = namespace
    return sq


class TestOutputTable:
    def test_simple(self):
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock())
        assert dsq.output_table == "test_ns.test_sq"

    def test_name_sanitized(self):
        dsq = DatabricksStagingQuery(_make_sq(name="my.query", namespace="ns"), MagicMock())
        assert dsq.output_table == "ns.my_query"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.side_effect = lambda t, n: [None] * n
        return spark

    def _get_cli_args(self, spark):
        return list(spark._jvm.ai.chronon.spark.Driver.main.call_args[0][0])

    def test_basic_args_passed_to_jvm(self):
        spark = self._make_spark()
        dsq = DatabricksStagingQuery(_make_sq(), spark)
        dsq._invoke_driver("/tmp/conf.json", "2026-04-01")
        spark._jvm.ai.chronon.spark.Driver.main.assert_called_once()

    def test_step_days_included_when_set(self):
        spark = self._make_spark()
        dsq = DatabricksStagingQuery(_make_sq(), spark)
        dsq._invoke_driver("/tmp/conf.json", "2026-04-01", step_days=7)
        args = self._get_cli_args(spark)
        assert "--step-days" in args
        assert "7" in args

    def test_no_step_days_omits_flag(self):
        spark = self._make_spark()
        dsq = DatabricksStagingQuery(_make_sq(), spark)
        dsq._invoke_driver("/tmp/conf.json", "2026-04-01")
        args = self._get_cli_args(spark)
        assert "--step-days" not in args

    def test_start_partition_included_when_set(self):
        spark = self._make_spark()
        dsq = DatabricksStagingQuery(_make_sq(), spark)
        dsq._invoke_driver("/tmp/conf.json", "2026-04-01", start_date="2026-03-01")
        args = self._get_cli_args(spark)
        assert "--start-partition" in args
        assert "2026-03-01" in args

    def test_raises_when_jar_not_on_classpath(self, spark, tmp_path):
        sq = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.sq__0", outputNamespace="data")
        )
        (tmp_path / "staging_query.json").write_text("{}")
        dsq = DatabricksStagingQuery(sq, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            dsq._invoke_driver(str(tmp_path / "staging_query.json"), "2026-01-01")


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        start_date=None,
        step_days=None,
        enable_auto_expand=True,
        force_overwrite=False,
        run_first_hole=True,
    )

    def _run(self, sq, spark, tmp_path, display_fn=None, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-01")
        dsq = DatabricksStagingQuery(sq, spark, tmp_dir=str(tmp_path), display_fn=display_fn)
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "staging_query.json")

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"),
            patch.object(dsq, "_invoke_driver") as mock_invoke,
        ):
            result = dsq.run(end_date, **run_kwargs)

        return result, expected_df, mock_invoke, conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(_make_sq(), spark, tmp_path)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-01", **self._DEFAULT_INVOKE_KWARGS)

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _ = self._run(_make_sq(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_sq")
        assert result is expected_df

    def test_display_fn_called_with_dataframe(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        result, expected_df, _, _ = self._run(_make_sq(), spark, tmp_path, display_fn=display_fn)
        display_fn.assert_called_once_with(expected_df)
        assert result is expected_df

    def test_display_fn_not_called_when_none(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        self._run(_make_sq(), spark, tmp_path, display_fn=None)
        display_fn.assert_not_called()

    def test_step_days_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(_make_sq(), spark, tmp_path, step_days=7)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "step_days": 7}
        )

    def test_start_date_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(
            _make_sq(), spark, tmp_path, start_date="2026-03-01"
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "start_date": "2026-03-01"}
        )

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        dsq = DatabricksStagingQuery(_make_sq(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"),
            patch.object(dsq, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            dsq.run("2026-04-01")

        mock_mkdtemp.assert_called_once()


class TestWidgets:
    def test_setup_widgets_calls_dbutils(self):
        dbutils = MagicMock()
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock(), dbutils=dbutils)
        dsq.setup_widgets(default_end_date="2026-04-01", default_start_date="2026-03-01")
        assert dbutils.widgets.text.call_count == 2

    def test_setup_widgets_raises_without_dbutils(self):
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dsq.setup_widgets()

    def test_get_widget_dates_returns_values(self):
        dbutils = MagicMock()
        dbutils.widgets.get.side_effect = lambda key: (
            "2026-04-01" if key == "end_date" else "2026-03-01"
        )
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock(), dbutils=dbutils)
        end_date, start_date = dsq.get_widget_dates()
        assert end_date == "2026-04-01"
        assert start_date == "2026-03-01"

    def test_get_widget_dates_blank_start_returns_none(self):
        dbutils = MagicMock()
        dbutils.widgets.get.side_effect = lambda key: "2026-04-01" if key == "end_date" else ""
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock(), dbutils=dbutils)
        end_date, start_date = dsq.get_widget_dates()
        assert end_date == "2026-04-01"
        assert start_date is None

    def test_get_widget_dates_raises_without_dbutils(self):
        dsq = DatabricksStagingQuery(_make_sq(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dsq.get_widget_dates()
