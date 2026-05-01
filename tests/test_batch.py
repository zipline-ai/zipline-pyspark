from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai.chronon.pyspark.jupyter.staging_query_executor import JupyterStagingQuery, _sanitize


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
        jsq = JupyterStagingQuery(_make_sq(), MagicMock())
        assert jsq.output_table == "test_ns.test_sq"

    def test_name_sanitized(self):
        jsq = JupyterStagingQuery(_make_sq(name="my.query", namespace="ns"), MagicMock())
        assert jsq.output_table == "ns.my_query"


class TestInvokeDriver:
    # Base arg count: subcommand + 5 flag pairs = 11
    _BASE_ARGS = 11

    def _make_spark(self):
        spark = MagicMock()
        spark.sparkContext._gateway.new_array.return_value = [None] * 20
        return spark

    def test_basic_args_passed_to_jvm(self):
        spark = self._make_spark()
        jsq = JupyterStagingQuery(_make_sq(), spark)
        jsq._invoke_driver("/tmp/conf.json", "2026-04-01")

        jvm = spark._jvm
        jvm.ai.chronon.spark.Driver.main.assert_called_once()

    def test_step_days_included_when_set(self):
        spark = self._make_spark()
        jsq = JupyterStagingQuery(_make_sq(), spark)
        jsq._invoke_driver("/tmp/conf.json", "2026-04-01", step_days=7)

        gateway = spark.sparkContext._gateway
        gateway.new_array.assert_called_once_with(spark._jvm.String, self._BASE_ARGS + 2)

    def test_no_step_days_omits_flag(self):
        spark = self._make_spark()
        jsq = JupyterStagingQuery(_make_sq(), spark)
        jsq._invoke_driver("/tmp/conf.json", "2026-04-01")

        gateway = spark.sparkContext._gateway
        gateway.new_array.assert_called_once_with(spark._jvm.String, self._BASE_ARGS)

    def test_start_partition_included_when_set(self):
        spark = self._make_spark()
        jsq = JupyterStagingQuery(_make_sq(), spark)
        jsq._invoke_driver("/tmp/conf.json", "2026-04-01", start_date="2026-03-01")

        gateway = spark.sparkContext._gateway
        gateway.new_array.assert_called_once_with(spark._jvm.String, self._BASE_ARGS + 2)


class TestRun:
    _DEFAULT_INVOKE_KWARGS = dict(
        start_date=None,
        step_days=None,
        enable_auto_expand=True,
        force_overwrite=False,
        run_first_hole=True,
    )

    def _run(self, sq, spark, tmp_path, **run_kwargs):
        end_date = run_kwargs.pop("end_date", "2026-04-01")
        jsq = JupyterStagingQuery(sq, spark, tmp_dir=str(tmp_path))
        expected_df = MagicMock()
        spark.table.return_value = expected_df
        conf_path = str(tmp_path / "staging_query.json")

        with (
            patch(
                "ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"
            ) as mock_compile,
            patch.object(jsq, "_invoke_driver") as mock_invoke,
        ):
            result = jsq.run(end_date, **run_kwargs)

        return result, expected_df, mock_invoke, mock_compile, conf_path

    def test_compile_called_with_correct_path(self, tmp_path):
        spark = MagicMock()
        _, _, _, mock_compile, conf_path = self._run(_make_sq(), spark, tmp_path)
        assert mock_compile.call_args[0][2] == conf_path

    def test_driver_called_with_defaults(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_sq(), spark, tmp_path)
        mock_invoke.assert_called_once_with(conf_path, "2026-04-01", **self._DEFAULT_INVOKE_KWARGS)

    def test_step_days_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(_make_sq(), spark, tmp_path, step_days=7)
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "step_days": 7}
        )

    def test_start_date_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_sq(), spark, tmp_path, start_date="2026-03-01"
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "start_date": "2026-03-01"}
        )

    def test_force_overwrite_forwarded(self, tmp_path):
        spark = MagicMock()
        _, _, mock_invoke, _, conf_path = self._run(
            _make_sq(), spark, tmp_path, force_overwrite=True
        )
        mock_invoke.assert_called_once_with(
            conf_path, "2026-04-01", **{**self._DEFAULT_INVOKE_KWARGS, "force_overwrite": True}
        )

    def test_returns_output_table_dataframe(self, tmp_path):
        spark = MagicMock()
        result, expected_df, _, _, _ = self._run(_make_sq(), spark, tmp_path)
        spark.table.assert_called_once_with("test_ns.test_sq")
        assert result is expected_df

    def test_default_tmp_dir_created(self, tmp_path):
        spark = MagicMock()
        spark.table.return_value = MagicMock()
        jsq = JupyterStagingQuery(_make_sq(), spark)

        with (
            patch("ai.chronon.pyspark.jupyter.session.ChrononSession.compile_to_file"),
            patch.object(jsq, "_invoke_driver"),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)) as mock_mkdtemp,
        ):
            jsq.run("2026-04-01")

        mock_mkdtemp.assert_called_once()


class TestInvokeDriverJarMissing:
    """Replicates the real-world TypeError when the Chronon JAR is absent from the classpath.

    Uses a plain SparkSession (no spark.jars set) so that Py4J returns a JavaPackage for
    ai.chronon.spark.Driver instead of the actual class.  Calling .main() on a JavaPackage
    raises TypeError: 'JavaPackage' object is not callable.
    """

    def test_raises_when_jar_not_on_classpath(self, spark, tmp_path):
        sq = SimpleNamespace(
            metaData=SimpleNamespace(name="gcp.test.sq__0", outputNamespace="data")
        )
        conf_path = str(tmp_path / "staging_query.json")
        (tmp_path / "staging_query.json").write_text("{}")

        jsq = JupyterStagingQuery(sq, spark)
        with pytest.raises(RuntimeError, match="ai.chronon.spark.Driver"):
            jsq._invoke_driver(conf_path, "2026-01-01")
