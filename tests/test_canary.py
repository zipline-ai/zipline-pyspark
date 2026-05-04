import json
import os
import sys

import pytest

CANARY_DIR = os.path.join(os.path.dirname(__file__), "canary")


class TestCompileToFileNameResolution:
    """Verify that compile_to_file resolves metaData.name via GC when it is missing.

    Uses the real dim_listings canary object so this exercises the actual GC referrer
    lookup, update_metadata, and serialization path — no mocks.
    """

    def test_dim_listings_name_in_compiled_json(self, tmp_path):
        from ai.chronon.pyspark.jupyter.session import ChrononSession

        if CANARY_DIR not in sys.path:
            sys.path.insert(0, CANARY_DIR)

        from staging_queries.gcp import exports

        sq = exports.dim_listings
        sq.metaData.name = None  # reset to simulate pre-compile state

        conf_path = str(tmp_path / "dim_listings.json")
        ChrononSession.compile_to_file(sq, CANARY_DIR, conf_path)

        with open(conf_path) as f:
            compiled = json.load(f)

        name = compiled["metaData"]["name"]
        assert name is not None
        assert "dim_listings" in name


class TestCanaryCompile:
    @pytest.mark.skip
    def test_gcp_dim_listings_matches_canary(self, tmp_path):
        """
        _compile_to_file on gcp/exports.dim_listings produces JSON identical to the canary file.
        """
        from gen_thrift.api.ttypes import StagingQuery as StagingQueryType

        from ai.chronon.cli.compile.parse_configs import from_file
        from ai.chronon.pyspark.batch import _compile_to_file

        if CANARY_DIR not in sys.path:
            sys.path.insert(0, CANARY_DIR)

        sq_dir = os.path.join(CANARY_DIR, "staging_queries")
        sq_file = os.path.join(sq_dir, "gcp", "exports.py")

        results = from_file(sq_file, StagingQueryType, sq_dir)
        obj = results["gcp.exports.dim_listings__0"]
        obj.metaData.sourceFile = os.path.relpath(sq_file, CANARY_DIR)

        conf_path = str(tmp_path / "dim_listings.json")
        _compile_to_file(obj, CANARY_DIR, conf_path)

        canary_path = os.path.join(
            CANARY_DIR, "compiled", "staging_queries", "gcp", "exports.dim_listings__0"
        )
        with open(conf_path) as f:
            actual = json.load(f)
        with open(canary_path) as f:
            expected = json.load(f)

        assert actual == expected
