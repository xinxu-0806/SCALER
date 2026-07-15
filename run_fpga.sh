#!/usr/bin/env bash
# Backward-compatible entry point for the HLS synthesis flow.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$repo_root/scripts/run_hls_synth.sh" "$@"
