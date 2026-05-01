from unittest.mock import MagicMock, patch

from ai.chronon.pyspark.batch import BatchStagingQuery, _sanitize


def _make_sq(name="test_sq", namespace="test_ns"):
    sq = MagicMock()
    sq.metaData.name = name
    sq.metaData.outputNamespace = namespace
    return sq


class TestSanitize:
    def test_replaces_dots_and_hyphens(self):
        assert _sanitize("foo.bar-baz") == "foo_bar_baz"

    def test_preserves_valid_chars(self):
        assert _sanitize("foo_Bar_123") == "foo_Bar_123"

    def test_none_passthrough(self):
        assert _sanitize(None) is None


class TestOutputTable:
    def test_simple(self):
        bsq = BatchStagingQuery(_make_sq(), MagicMock())
        assert bsq.output_table == "test_ns.test_sq"

    def test_name_sanitized(self):
        bsq = BatchStagingQuery(_make_sq(name="my.query", namespace="ns"), MagicMock())
        assert bsq.output_table == "ns.my_query"


class TestInvokeDriver:
    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.return_value = [None] * 10
        return spark

    def test_basic_args_passed_to_jvm(self):
        spark = self._make_spark()
        bsq = BatchStagingQuery(_make_sq(), spark)
        bsq._invoke_driver("/tmp/conf.json", "2026-04-01", None)

        jvm = spark._jvm
        jvm.ai.chronon.spark.batch.StagingQuery.main.assert_called_once()

    def test_step_days_included_when_set(self):
        spark = self._make_spark()
        bsq = BatchStagingQuery(_make_sq(), spark)

        with patch.object(spark._jvm.ai.chronon.spark.batch.StagingQuery, "main"):
            bsq._invoke_driver("/tmp/conf.json", "2026-04-01", 7)

        gateway = spark.sparkContext._gateway
        # 6 args: --conf-path, path, --end-date, date, --step-days, 7
        gateway.new_array.assert_called_once_with(spark._jvm.String, 6)

    def test_no_step_days_omits_flag(self):
        spark = self._make_spark()
        bsq = BatchStagingQuery(_make_sq(), spark)
        bsq._invoke_driver("/tmp/conf.json", "2026-04-01", None)

        gateway = spark.sparkContext._gateway
        # 4 args: --conf-path, path, --end-date, date
        gateway.new_array.assert_called_once_with(spark._jvm.String, 4)


class TestRun:
    def _run(self, sq, spark, tmp_path, end_date="2026-04-01", step_days=None):
        bsq = BatchStagingQuery(sq, spark, tmp_dir=str(tmp_path))
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "staging_query.json")

        with (
            patch("ai.chronon.pyspark.batch._compile_to_file") as mock_compile,
            patch.object(bsq, "_invoke_driver") as mock_invoke,
        ):
            result = bsq.run(end_date, step_days=step_days)

        return result, expected_df, mock_invoke, mock_compile, conf_path

    def test_compile_called_with_correct_path(self, tmp_path):
        spark = MagicMock()
        sq = _make_sq()
        _, _, _, mock_compile, conf_path = self._run(sq, spark, tmp_path)
        assert mock_compile.call_args[0][2] == conf_path

    def test_driver_called_with_correct_args(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_sq(), spark, tmp_path)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-01", None)

    def test_step_days_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_sq(), spark, tmp_path, step_days=7)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-01", 7)

    def test_yyyymmdd_date_normalized(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_sq(), spark, tmp_path, end_date="20260401"
        )
        mock_invoke.assert_called_once_with(conf_path, "2026-04-01", None)

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _, _ = self._run(_make_sq(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_sq")
        assert result is expected_df

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        bsq = BatchStagingQuery(_make_sq(), spark)

        with (
            patch("ai.chronon.pyspark.batch._compile_to_file"),
            patch.object(bsq, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            bsq.run("2026-04-01")

        mock_mkdtemp.assert_called_once()
