#!/usr/bin/env bash
# Public-compatible entry point: synthesize/link and publish the bitstream.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$repo_root/scripts/generate_bitstream.sh" "$@"
