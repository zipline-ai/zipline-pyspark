# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install for development (includes all lint/test tools)
pip install -e ".[dev]"
pre-commit install

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_jupyter.py::TestJupyterStagingQueryChunking::test_single_day_no_start -v

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Format code
black --line-length 100 src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/

# Run all pre-commit hooks on all files (skipping the pytest hook)
SKIP=pytest pre-commit run --all-files

# Build the package
python -m build --no-isolation
```

## Architecture

The package lives under `src/ai/chronon/pyspark/` using a PEP 420 implicit namespace package (`ai.chronon` is shared with the upstream Chronon project).

**Core pattern:** each public class wraps a Zipline entity (e.g. a `StagingQuery` config object) and a live `SparkSession`, exposing a single `.run(start_date, end_date, step_days)` method. Internally, `.run` iterates over date windows, renders the SQL template for each window via `_render_query`, executes it against the session, and unions the resulting DataFrames.

**Current state:** only `JupyterStagingQuery` is implemented (`jupyter.py`). `JupyterGroupBy` and `JupyterJoin` are referenced in the README/usage examples but not yet written. A `databricks.py` backend (same interface, Databricks-specific display/widget integration) is planned.

**Key internals in `jupyter.py`:**
- `_parse_date` — accepts `YYYYMMDD` or `YYYY-MM-DD`
- `_render_query` — substitutes `{{ start_date }}` / `{{ end_date }}` placeholders (with optional function-call syntax) in SQL strings
- `JupyterStagingQuery.run` — date-range chunking loop; `enable_auto_expand=True` extends `start_date` back by one `step_days`

**Test harness (`tests/`):**
- `conftest.py` — session-scoped `SparkSession` fixture (local mode, `spark.driver.bindAddress=127.0.0.1` required on macOS/CI)
- Zipline entity stubs are simple ad-hoc objects with `.query` and `.setups` attributes — no `zipline-ai` import needed in tests

**Dependency note:** `zipline-ai` is an internal Zipline package and is not available on public PyPI. CI installs the package with `--no-deps` and supplies `pyspark` explicitly to avoid this.

## Code style

- Line length: 100 (black + flake8 + isort all configured to 100)
- `isort` profile: `black`; `ai.*` imports are `known_first_party`
- Docstring warnings (`D1xx`) are suppressed project-wide; all `D` rules are suppressed in `tests/`
