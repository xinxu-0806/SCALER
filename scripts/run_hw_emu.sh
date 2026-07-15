#!/usr/bin/env bash
# Link an hw_emu xclbin, generate emconfig.json, and run one matrix.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

matrix_file="${1:-$REPO_ROOT/data/add20.mtx}"
require_file "$matrix_file"
matrix_file="$(cd "$(dirname "$matrix_file")" && pwd)/$(basename "$matrix_file")"
require_file "$HBM_CONFIG"

if [[ ! -f "$XO_FILE" ]]; then
  "$SCRIPT_DIR/run_hls_synth.sh"
fi

emu_dir="$BUILD_DIR/hw_emu"
xclbin="$emu_dir/${TOP_FUNCTION}_${PLATFORM}.hw_emu.xclbin"
mkdir -p "$emu_dir"

v++ -l --target hw_emu \
  --platform "$PLATFORM" \
  --config "$HBM_CONFIG" \
  --temp_dir "$emu_dir/link_tmp" \
  "$XO_FILE" \
  -o "$xclbin"

emconfigutil --platform "$PLATFORM" --nd 1 --od "$emu_dir"
make -C "$REPO_ROOT" sw_emu

cd "$emu_dir"
XCL_EMULATION_MODE=hw_emu "$BUILD_DIR/host/lu_solver_host_emu" \
  --bitstream="$xclbin" \
  --matrix_file="$matrix_file"
