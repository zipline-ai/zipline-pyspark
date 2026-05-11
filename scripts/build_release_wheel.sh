#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/build_release_wheel.sh <version-or-tag>"
  echo "Example: scripts/build_release_wheel.sh v0.2.0"
  exit 1
fi

version="${1#v}"
expected_wheel="zipline_pyspark-${version}-py3-none-any.whl"

python -m pip install --upgrade build "setuptools_scm[toml]>=8"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZIPLINE_PYSPARK="${version}" python -m build --wheel --no-isolation

if [[ ! -f "dist/${expected_wheel}" ]]; then
  echo "${expected_wheel} not found"
  exit 1
fi

echo "Built dist/${expected_wheel}"
