import re
from datetime import datetime, timedelta
from functools import reduce
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def _parse_date(ds: str) -> datetime:
    """Accept YYYYMMDD or YYYY-MM-DD and return a datetime."""
    ds = ds.strip()
    if re.fullmatch(r"\d{8}", ds):
        return datetime.strptime(ds, "%Y%m%d")
    return datetime.strptime(ds, "%Y-%m-%d")


def _render_query(query: str, start_date: str, end_date: str) -> str:
    """Substitute {{ start_date }} and {{ end_date }} placeholders in the query."""
    result = re.sub(r"\{\{\s*start_date(?:\([^)]*\))?\s*\}\}", start_date, query)
    result = re.sub(r"\{\{\s*end_date(?:\([^)]*\))?\s*\}\}", end_date, result)
    return result


class JupyterStagingQuery:
    """Executes a Chronon StagingQuery directly against a PySpark session.

    Intended for interactive use in Jupyter notebooks where running a full
    spark-submit job is impractical.  Handles date-range chunking and
    query template rendering; the full upstream auto-expand logic from
    the Chronon Spark Driver is not replicated here.
    """

    def __init__(self, staging_query, spark: SparkSession):
        self.staging_query = staging_query
        self.spark = spark

    def run(
        self,
        end_date: str,
        step_days: int = 1,
        enable_auto_expand: bool = False,
        start_date: Optional[str] = None,
    ) -> DataFrame:
        """Execute the staging query and return the result as a single DataFrame.

        Dates can be supplied in YYYYMMDD or YYYY-MM-DD format.

        When step_days > 1 each chunk is executed separately and the results
        are union-ed together.  When enable_auto_expand is True the start date
        is extended back by one step from the nominal start so that the first
        step picks up any newly available upstream partitions.

        Args:
            end_date: Inclusive end date.
            step_days: Days per execution step.
            enable_auto_expand: Extend the date range one extra step at the
                start to incorporate newly available upstream data.
            start_date: Explicit start date.  When omitted the query is run
                as a single step covering end_date only.
        """
        end_dt = _parse_date(end_date)

        if start_date is not None:
            start_dt = _parse_date(start_date)
        else:
            # Default: single-day run ending on end_date
            start_dt = end_dt

        if enable_auto_expand:
            start_dt = start_dt - timedelta(days=step_days)

        # Run setup statements (UDF registrations, etc.)
        for stmt in self.staging_query.setups or []:
            self.spark.sql(stmt)

        # Build list of (step_start, step_end) windows
        windows = []
        current = start_dt
        while current <= end_dt:
            step_end = min(current + timedelta(days=step_days - 1), end_dt)
            windows.append((current, step_end))
            current = step_end + timedelta(days=1)

        dfs = []
        for step_start, step_end in windows:
            query = _render_query(
                self.staging_query.query,
                step_start.strftime("%Y-%m-%d"),
                step_end.strftime("%Y-%m-%d"),
            )
            dfs.append(self.spark.sql(query))

        if not dfs:
            return self.spark.createDataFrame([], StructType([]))

        # Union all step results; duplicates across boundaries are intentionally
        # preserved so callers can deduplicate by partition key if needed.
        return reduce(DataFrame.union, dfs)
