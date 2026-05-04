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

# Run integration tests (requires a cloud JAR in tests/resources/jars/cloud_<cloud>.jar)
pytest tests/integration/ -m integration --cloud aws   # or gcp / azure
pytest tests/integration/ -m integration               # auto-selects first available JAR

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

**`batch.py` — `_compile_to_file(staging_query, chronon_root, conf_path)`:**
- Requires `staging_query.metaData.name` and `.team` to be set before calling.
- Calls `ai.chronon.cli.compile.parse_teams.load_teams(chronon_root)` (reads `teams.py` from `chronon_root`) then `update_metadata` (namespace propagation, team conf merging), then serializes with `ai.chronon.cli.compile.serializer.thrift_simple_json`.
- Does NOT set `metaData.sourceFile` — that is set by the CLI `from_folder` loop after `from_file`.
- `BatchStagingQuery` wraps this: compiles to a temp dir then calls the Scala driver via Py4J.

**Compile pipeline (how the CLI produces canary files):**
1. `parse_configs.from_file(file_path, cls, input_dir)` — imports the module, finds all `cls` instances, deep-copies each, sets `metaData.name = "{team}.{module}.{var}__{version}"` and `metaData.team = module.split(".")[0]`. Requires `chronon_root` (parent of `input_dir`) on `sys.path`.
2. `parse_teams.update_metadata(obj, teams_dict)` — merges team namespace/conf/env onto the object.
3. `airflow_helpers.set_airflow_deps(obj)` — **no-op for StagingQuery** (airflow deps are written into `metaData.customJson` at object-creation time by the `StagingQuery` wrapper in `ai.chronon.staging_query`).
4. Set `obj.metaData.sourceFile = os.path.relpath(file_path, chronon_root)`.
5. Serialize with `thrift_simple_json`.

**Type distinction:** `ai.chronon.types.StagingQuery` is the Python wrapper/constructor (sets `customJson`, derives team from caller filename via `utils._get_team_from_caller()`). `gen_thrift.api.ttypes.StagingQuery` is the raw thrift type used for `isinstance` checks.

**Test harness (`tests/`):**
- `conftest.py` — session-scoped `SparkSession` fixture (local mode, `spark.driver.bindAddress=127.0.0.1` required on macOS/CI)
- Zipline entity stubs are simple ad-hoc objects with `.query` and `.setups` attributes — no `zipline-ai` import needed in tests
- `tests/canary/` — real Zipline config objects (group_bys, joins, staging_queries) organized by cloud provider (`aws/`, `azure/`, `gcp/`), with pre-compiled golden outputs in `tests/canary/compiled/`. Used for compile-pipeline regression tests (`test_canary.py`).
- To use `parse_configs.from_file` in tests, add `tests/canary` to `sys.path` first so `staging_queries.gcp.exports` etc. are importable as top-level modules.

**Dependency note:** `zipline-ai` is an internal Zipline package and is not available on public PyPI. CI installs the package with `--no-deps` and supplies `pyspark` explicitly to avoid this.

## Code style

- Line length: 100 (black + flake8 + isort all configured to 100)
- `isort` profile: `black`; `ai.*` imports are `known_first_party`
- Docstring warnings (`D1xx`) are suppressed project-wide; all `D` rules are suppressed in `tests/`
