#!/usr/bin/env bash
# Run TAPA/Vitis HLS synthesis and emit a hardware .xo object.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

clock_period="${CLOCK_PERIOD:-7.69}"
jobs="${JOBS:-16}"
work_dir="$FPGA_DIR/${TOP_FUNCTION}_${PLATFORM}.tapa_work"
log_file="$FPGA_DIR/tapa_compile_output.log"

require_file "$KERNEL_SOURCE"
require_file "$HBM_CONFIG"
mkdir -p "$FPGA_DIR"

echo "[HLS] platform=$PLATFORM clock=${clock_period}ns jobs=$jobs"
echo "[HLS] xo=$XO_FILE"

tapa -w "$work_dir" compile \
  -f "$KERNEL_SOURCE" \
  -t "$TOP_FUNCTION" \
  -p "$PLATFORM" \
  --clock-period "$clock_period" \
  -j "$jobs" \
  --enable-synth-util \
  --gen-ab-graph \
  --keep-hls-work-dir \
  -o "$XO_FILE" \
  2>&1 | tee "$log_file"
