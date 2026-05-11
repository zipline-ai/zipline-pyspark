# zipline-pyspark

[![CI](https://github.com/zipline-ai/zipline-pyspark/actions/workflows/ci.yml/badge.svg)](https://github.com/zipline-ai/zipline-pyspark/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

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

## Setup

### Prerequisites

- Python 3.11+
- PySpark 3.5+
- A running Spark session (local or remote)
- `zipline-ai` package (installed automatically as a dependency)

### Databricks limitations

The Databricks notebook executors invoke Chronon's Scala batch driver through the Spark
driver JVM. Databricks serverless compute does not support direct access to the underlying
driver JVM via `sparkContext`, so these executors are not compatible with serverless
notebook compute. Use classic Databricks compute with Dedicated access mode, and attach the
Chronon batch JAR to the cluster.

### Development install

```bash
git clone https://github.com/zipline-ai/zipline-pyspark.git
cd zipline-pyspark
pip install -e ".[dev]"
pre-commit install
```

### Running tests

```bash
pytest tests/ -v
```

Coverage is enforced automatically (configured in `pyproject.toml`). To skip the coverage check locally:

```bash
pytest tests/ -v --no-cov
```

## Release Flow

Each PR must carry exactly one semantic version label: `Semver-Major`, `Semver-Minor`, or
`Semver-Patch`. Release Drafter uses those labels to keep a draft GitHub release up to date
and determine the next release version.

Publishing that GitHub release builds the `zipline-pyspark` wheel with the release tag as
the package version and publishes it to PyPI through trusted publishing. To smoke-test the
same wheel build locally:

```bash
scripts/build_release_wheel.sh v0.2.0
```

## Roadmap

### 1. Implement `jupyter` and `databricks` backends

- **`jupyter.py`** — complete `JupyterGroupBy` and `JupyterJoin` to match the existing `JupyterStagingQuery` pattern: date-range chunking, setup-statement execution, and union of per-step DataFrames.
- **`databricks.py`** — a parallel set of classes (`DatabricksGroupBy`, `DatabricksJoin`, `DatabricksStagingQuery`) that adapt the same interface for Databricks notebooks: `dbutils`-aware progress display, widget-based date inputs, and Databricks `displayHTML` / `display` integration instead of plain DataFrame returns.
- **Shared base layer** — extract common chunking and template-rendering logic into an internal `_base.py` so both backends stay in sync without code duplication.

### 2. Distribution

- Publish to PyPI under `zipline-pyspark` and document the `%pip install zipline-pyspark` notebook workflow.

## Credits

This project is the Zipline counterpart of the original implementation contributed to the open-source [Chronon](https://github.com/airbnb/chronon) project by Airbnb. See the original pull request for reference: [airbnb/chronon#981](https://github.com/airbnb/chronon/pull/981/files).

## License

See [LICENSE](LICENSE).
