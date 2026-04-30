from datetime import datetime

import pytest

from ai.chronon.pyspark.jupyter import (
    JupyterStagingQuery,
    _parse_date,
    _render_query,
)


class TestParseDate:
    def test_yyyymmdd(self):
        assert _parse_date("20260401") == datetime(2026, 4, 1)

    def test_yyyy_mm_dd(self):
        assert _parse_date("2026-04-01") == datetime(2026, 4, 1)

    def test_strips_whitespace(self):
        assert _parse_date("  2026-04-01  ") == datetime(2026, 4, 1)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")


class TestRenderQuery:
    def test_start_date_placeholder(self):
        q = _render_query(
            "SELECT * FROM t WHERE ds >= '{{ start_date }}'",
            "2026-04-01",
            "2026-04-07",
        )
        assert "2026-04-01" in q
        assert "{{ start_date }}" not in q

    def test_end_date_placeholder(self):
        q = _render_query(
            "SELECT * FROM t WHERE ds <= '{{end_date}}'",
            "2026-04-01",
            "2026-04-07",
        )
        assert "2026-04-07" in q
        assert "{{end_date}}" not in q

    def test_both_placeholders(self):
        q = _render_query("{{ start_date }} to {{ end_date }}", "2026-04-01", "2026-04-07")
        assert q == "2026-04-01 to 2026-04-07"

    def test_placeholder_with_function_syntax(self):
        q = _render_query("{{start_date(some_arg)}}", "2026-04-01", "2026-04-07")
        assert q == "2026-04-01"

    def test_no_placeholders_unchanged(self):
        query = "SELECT * FROM t"
        assert _render_query(query, "2026-04-01", "2026-04-07") == query


class TestJupyterStagingQueryChunking:
    def _sq(self, query, setups=None):
        class _FakeSQ:
            pass

        obj = _FakeSQ()
        obj.query = query
        obj.setups = setups or []
        return obj

    def test_single_day_no_start(self, spark):
        result = JupyterStagingQuery(self._sq("SELECT '{{ end_date }}' AS ds"), spark).run(
            end_date="2026-04-01"
        )
        assert result.count() == 1

    def test_multi_step_step1_produces_union(self, spark):
        result = JupyterStagingQuery(self._sq("SELECT '{{ start_date }}' AS ds"), spark).run(
            start_date="2026-04-01", end_date="2026-04-03", step_days=1
        )
        assert result.count() == 3

    def test_step_days_chunking(self, spark):
        result = JupyterStagingQuery(self._sq("SELECT '{{ start_date }}' AS ds"), spark).run(
            start_date="2026-04-01", end_date="2026-04-04", step_days=2
        )
        assert result.count() == 2

    def test_setup_statements_executed(self, spark):
        sq = self._sq(
            query="SELECT x FROM _setup_test WHERE '{{ start_date }}' = '{{ start_date }}'",
            setups=["CREATE OR REPLACE TEMP VIEW _setup_test AS SELECT 42 AS x"],
        )
        result = JupyterStagingQuery(sq, spark).run(end_date="2026-04-01")
        assert result.count() == 1
        assert result.collect()[0]["x"] == 42

    def test_enable_auto_expand_extends_start(self, spark):
        # With auto-expand, the start is pushed back by step_days, producing one extra step.
        result = JupyterStagingQuery(self._sq("SELECT '{{ start_date }}' AS ds"), spark).run(
            start_date="2026-04-02",
            end_date="2026-04-03",
            step_days=1,
            enable_auto_expand=True,
        )
        # 2026-04-01, 2026-04-02, 2026-04-03 → 3 rows
        assert result.count() == 3
