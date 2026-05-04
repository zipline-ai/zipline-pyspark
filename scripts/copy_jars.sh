#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <chronon_oss_repo>"
    echo "  chronon_oss_repo  Path to the chronon OSS repository (must have pre-built JARs in out/)"
    exit 1
}

[[ $# -eq 1 ]] || usage

CHRONON_REPO="$1"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/tests/resources/jars"

for cloud in aws azure gcp; do
    src="$CHRONON_REPO/out/cloud_${cloud}/assembly.dest/out.jar"
    dst="$DEST_DIR/cloud_${cloud}.jar"
    if [[ ! -f "$src" ]]; then
        echo "WARNING: $src not found, skipping $cloud" >&2
        continue
    fi
    cp "$src" "$dst"
    echo "Copied cloud_${cloud}: $src -> $dst"
done
