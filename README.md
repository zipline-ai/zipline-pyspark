# zipline-pyspark

Execution utilities to help with iteration and UX when working with [Zipline](https://zipline.ai) in notebook environments.

> **Status:** Work in progress.

## Overview

`zipline-pyspark` provides thin wrappers around Zipline's core entities — `GroupBy`, `Join`, and `StagingQuery` — that are designed to run interactively inside a Jupyter (or compatible) notebook against a live PySpark session. The goal is to make it easy to prototype, debug, and iterate on Zipline feature definitions without leaving the notebook.

## Installation

```bash
%pip install zipline-pyspark
```

## Usage

```python
from ai.chronon.pyspark.jupyter import JupyterGroupBy, JupyterJoin, JupyterStagingQuery

# Run a GroupBy over a date range, chunked by day
result_df = JupyterGroupBy(my_group_by, spark).run(
    start_date="2026-04-01",
    end_date="2026-04-07",
    step_days=1,
)

# Run a Join
result_df = JupyterJoin(my_join, spark).run(
    start_date="2026-04-01",
    end_date="2026-04-07",
)

# Run a StagingQuery
result_df = JupyterStagingQuery(my_staging_query, spark).run(
    end_date="2026-04-07",
    start_date="2026-04-01",
    step_days=1,
)
```

## Credits

This project is the Zipline counterpart of the original implementation contributed to the open-source [Chronon](https://github.com/airbnb/chronon) project by Airbnb. See the original pull request for reference: [airbnb/chronon#981](https://github.com/airbnb/chronon/pull/981/files).

## License

See [LICENSE](LICENSE).
