#!/usr/bin/env bash
# Build and run TAPA software emulation for one Matrix Market matrix.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

matrix_file="${1:-$REPO_ROOT/data/add20.mtx}"
require_file "$matrix_file"

make -C "$REPO_ROOT" sw_emu
exec "$BUILD_DIR/host/lu_solver_host_emu" --matrix_file="$matrix_file"
