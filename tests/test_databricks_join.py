from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.databricks.join_executor import DatabricksJoin


def _make_join(name="test_join", namespace="test_ns"):
    j = MagicMock()
    j.metaData.name = name
    j.metaData.outputNamespace = namespace
    return j


class TestOutputTable:
    def test_simple(self):
        dj = DatabricksJoin(_make_join(), MagicMock())
        assert dj.output_table == "test_ns.test_join"

    def test_name_sanitized(self):
        dj = DatabricksJoin(_make_join(name="my.join-v1", namespace="ns"), MagicMock())
        assert dj.output_table == "ns.my_join_v1"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.side_effect = lambda t, n: [None] * n
        return spark

    def _get_cli_args(self, spark):
        return list(spark._jvm.ai.chronon.spark.Driver.main.call_args[0][0])

    def test_subcommand_is_join(self):
        spark = self._make_spark()
        dj = DatabricksJoin(_make_join(), spark)
        dj._invoke_driver("/tmp/conf.json", "2026-04-07")
        args = self._get_cli_args(spark)
        assert "join" in args

    def test_start_partition_included_when_set(self):
        spark = self._make_spark()
        dj = DatabricksJoin(_make_join(), spark)
        dj._invoke_driver("/tmp/conf.json", "2026-04-07", start_date="2026-04-01")
        args = self._get_cli_args(spark)
        assert "--start-partition" in args
        assert "2026-04-01" in args

    def test_start_partition_omitted_when_none(self):
        spark = self._make_spark()
        dj = DatabricksJoin(_make_join(), spark)
        dj._invoke_driver("/tmp/conf.json", "2026-04-07")
        args = self._get_cli_args(spark)
        assert "--start-partition" not in args

    def test_raises_when_jar_not_on_classpath(self, spark, tmp_path):
        join = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.join__0", outputNamespace="data")
        )
        (tmp_path / "join.json").write_text("{}")
        dj = DatabricksJoin(join, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            dj._invoke_driver(str(tmp_path / "join.json"), "2026-01-01")


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        start_date=None,
        run_first_hole=True,
    )

    def _run(self, join_obj, spark, tmp_path, display_fn=None, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-07")
        dj = DatabricksJoin(join_obj, spark, tmp_dir=str(tmp_path), display_fn=display_fn)
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "join.json")

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_join_to_file"),
            patch.object(dj, "_invoke_driver") as mock_invoke,
        ):
            result = dj.run(end_date, **run_kwargs)

        return result, expected_df, mock_invoke, conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(_make_join(), spark, tmp_path)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-07", **self._DEFAULT_INVOKE_KWARGS)

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _ = self._run(_make_join(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_join")
        assert result is expected_df

    def test_display_fn_called_with_dataframe(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        result, expected_df, _, _ = self._run(_make_join(), spark, tmp_path, display_fn=display_fn)
        display_fn.assert_called_once_with(expected_df)
        assert result is expected_df

    def test_display_fn_not_called_when_none(self, tmp_path):
        spark = MagicMock()
        display_fn = MagicMock()
        self._run(_make_join(), spark, tmp_path, display_fn=None)
        display_fn.assert_not_called()

    def test_start_date_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, conf_path = self._run(
            _make_join(), spark, tmp_path, start_date="2026-04-01"
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-07", **{**self._DEFAULT_INVOKE_KWARGS, "start_date": "2026-04-01"}
        )

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        dj = DatabricksJoin(_make_join(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_join_to_file"),
            patch.object(dj, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            dj.run("2026-04-07")

        mock_mkdtemp.assert_called_once()


class TestWidgets:
    def test_setup_widgets_calls_dbutils(self):
        dbutils = MagicMock()
        dj = DatabricksJoin(_make_join(), MagicMock(), dbutils=dbutils)
        dj.setup_widgets(default_end_date="2026-04-07", default_start_date="2026-04-01")
        assert dbutils.widgets.text.call_count == 2

    def test_setup_widgets_raises_without_dbutils(self):
        dj = DatabricksJoin(_make_join(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dj.setup_widgets()

    def test_get_widget_dates_returns_values(self):
        dbutils = MagicMock()
        dbutils.widgets.get.side_effect = lambda key: (
            "2026-04-07" if key == "end_date" else "2026-04-01"
        )
        dj = DatabricksJoin(_make_join(), MagicMock(), dbutils=dbutils)
        end_date, start_date = dj.get_widget_dates()
        assert end_date == "2026-04-07"
        assert start_date == "2026-04-01"

    def test_get_widget_dates_blank_start_returns_none(self):
        dbutils = MagicMock()
        dbutils.widgets.get.side_effect = lambda key: "2026-04-07" if key == "end_date" else ""
        dj = DatabricksJoin(_make_join(), MagicMock(), dbutils=dbutils)
        end_date, start_date = dj.get_widget_dates()
        assert end_date == "2026-04-07"
        assert start_date is None

    def test_get_widget_dates_raises_without_dbutils(self):
        dj = DatabricksJoin(_make_join(), MagicMock())
        with pytest.raises(RuntimeError, match="dbutils"):
            dj.get_widget_dates()
